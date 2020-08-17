import numpy as np
import gym
import random
import time
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

@functools.lru_cache(maxsize=None)
def explore_rate(eps):
    max_rate = 1
    min_rate = 0.01
    rate = min_rate + (max_rate - min_rate) * np.exp(-0.001 * eps)
    # rate = min_rate + (max_rate - min_rate) * 0.5
    return rate

def test_should_explore():
    current_eps = 0
    for i in range(500):
        current_eps += 1
        print(should_explore(current_eps))


class ReplayMem():
    def __init__(s, capacity):
        s.capa = capacity
        s.memo = []
        s.count = 0
    def push(s, experience):
        if len(s.memo) >= s.capa:
            s.memo[s.count % s.capa] = experience
        else:
            s.memo.append(experience)
        s.count += 1
    def sample(s, batch=1):
        return random.sample(s.memo, batch)[0]

def state_normalize(state):
    one_hot = torch.zeros(16)
    one_hot[state] = 1.0
    return one_hot


def get_action(network, state):
    state = state_normalize(state)
    return network(state).argmax().item()

def get_q_value(network, state ,action):
    state = state_normalize(state)
    return network(state).select(0, action)

def get_max_q_value(network, state):
    state = state_normalize(state)
    return network(state).max()

class DQN(nn.Module):
    def __init__(s):
        super().__init__()
        s.fc1 = nn.Linear(16, 10)
        s.fc2 = nn.Linear(10, 15)
        s.out = nn.Linear(15, 4)
    def forward(s, state):
        t = F.relu(s.fc1(state))
        t = F.relu(s.fc2(t))
        t = s.out(t)
        return t

env = gym.make("FrozenLake-v0")
current_eps = 0
memo = ReplayMem(100)
gamma = 0.9
lr = 0.01

policy_net = DQN()
target_net = DQN()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)


def copy_params_from_policy_to_target():
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

def similate_and_store_sample(env, memo, cur_eps, cur_state):
    # Simulator
    # get action
    action = env.action_space.sample() if explore_rate(cur_eps) > random.uniform(0, 1) else get_action(policy_net, cur_state)
    # take action
    new_state, reward, done, _ = env.step(action)
    # store to display memory
    memo.push((cur_state, action, reward, new_state))
    return (new_state, reward, done)

def train_policy_net_from_memo():
    # Training network
    # 1. sample from memo
    src_state, action, reward, tar_state = memo.sample()
    # 2. Get q-value for specified action using policy net
    predicted_q_value = get_q_value(policy_net, src_state, action)
    # 3. Get max q-value for next state using target net
    actual_q_value = (gamma * get_max_q_value(target_net, tar_state)) + reward
    # 4. Get loss by substracting, then backward
    loss = F.mse_loss(predicted_q_value, actual_q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def run_eps():
    global current_eps
    current_eps += 1
    state_before_action = env.reset()
    done = False
    step = 0
    while not done:
        step += 1
        state_before_action, reward, done = similate_and_store_sample(env, memo, current_eps, state_before_action)
        train_policy_net_from_memo()
    print(f"Eps {current_eps} done, total step: {step}, reward: {reward}")


def reset_target_net_and_run_n_eps(n = 10):
    copy_params_from_policy_to_target()
    for i in range(n):
        run_eps()


def play_and_render_by_policy_net():
    state = env.reset()
    done = False
    while not done:
        action = get_action(policy_net, state)
        state, reward, done, _ = env.step(action)
        env.render()

def print_all_q_values():
    for i in range(15):
        val = get_max_q_value(policy_net, i)
        print(f"state: {i}, value: {val}")

def print_all_action():
    for i in range(15):
        a = get_action(policy_net, i)
        print(f"state: {i}, action: {a}")
        
