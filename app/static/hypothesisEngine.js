export const hypState = {
values:[42,31,27]
}

export function normalize(values){
const sum = values.reduce((a,b)=>a+b,0)
return values.map(v=>Math.round(v/sum*100))
}

export function evolveHypothesis(){

let vals=[...hypState.values]

vals=vals.map(v=>Math.max(8,v+(Math.random()-0.5)*10))

if(Math.random()<0.2){
const winner=Math.floor(Math.random()*vals.length)
vals[winner]+=15
}

vals=normalize(vals)

hypState.values=vals

return vals
}