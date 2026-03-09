import {evolveHypothesis,hypState} from "./hypothesisEngine.js"
import {pushHistory,drawHistory} from "./historyChart.js"

function updateUI(vals){

vals.forEach((v,i)=>{

document.getElementById("hypFill"+i).style.width=v+"%"
document.getElementById("hypVal"+i).textContent=v+"%"

})

const idx=vals.indexOf(Math.max(...vals))

const labels=[
"BULLISH TREND",
"VOLATILITY EVENT",
"MEAN REVERSION"
]

document.getElementById("llmAnswer").textContent=labels[idx]

}

function tick(){

const vals=evolveHypothesis()

pushHistory(vals)

drawHistory()

updateUI(vals)

}

setInterval(tick,1400)

tick()
