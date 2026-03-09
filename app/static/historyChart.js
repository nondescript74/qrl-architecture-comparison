export const history={
max:36,
series:[
[42],
[31],
[27]
]
}

export function pushHistory(vals){

vals.forEach((v,i)=>{

history.series[i].push(v)

if(history.series[i].length>history.max)
history.series[i].shift()

})

}

export function drawHistory(){

const canvas=document.getElementById("hypHistoryChart")
const ctx=canvas.getContext("2d")

ctx.clearRect(0,0,canvas.width,canvas.height)

const colors=["#0072ff","#00875a","#ff8f00"]

history.series.forEach((series,i)=>{

ctx.strokeStyle=colors[i]
ctx.beginPath()

series.forEach((v,x)=>{

const px=x*(canvas.width/history.max)
const py=canvas.height-(v/100*canvas.height)

if(x===0)ctx.moveTo(px,py)
else ctx.lineTo(px,py)

})

ctx.stroke()

})

}