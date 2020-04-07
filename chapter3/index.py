import json, gzip

def q20():
  with gzip.open('jawiki-country.json.gz', mode='rt') as f:
    for line in filter(lambda x: json.loads(x)['title'] == "イギリス" ,f.readlines()):
      print(line, end='')

q20()