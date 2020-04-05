function words(str: string){
  return str.split(' ').map(item => {
    return item.replace('.', '')
  })
}

type StringIntMap = {[name: string]: number}

function f4(){
  const theWords = words('Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.')
  const oneCharIds = [1, 5, 6, 7, 8, 9, 15, 16, 19].map(item => item - 1);
  const resultWords : Array<Array<string | number>> = theWords.map((word, index) => {
    if(oneCharIds.indexOf(index) != -1){
      return [word.substr(0, 1), index + 1]
    }else{
      return [word.substr(0, 2), index + 1]
    }
  })

  let stringIntMap : StringIntMap = {};
  return resultWords.reduce((map,[key, val]) => {
    map[key as string] = val as number;
    return map;
  }, stringIntMap)
}

//console.log(f4())
