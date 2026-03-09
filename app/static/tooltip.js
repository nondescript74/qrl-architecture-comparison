const tooltip = document.getElementById("tooltip")
const tTitle = document.getElementById("tooltipTitle")
const tText = document.getElementById("tooltipText")

document.querySelectorAll(".help-target").forEach(el=>{

el.addEventListener("mouseenter",(e)=>{
tTitle.textContent = el.dataset.tipTitle || ""
tText.textContent = el.dataset.tipText || ""
tooltip.style.display="block"
})

el.addEventListener("mousemove",(e)=>{
tooltip.style.left = e.clientX+12+"px"
tooltip.style.top = e.clientY+12+"px"
})

el.addEventListener("mouseleave",()=>{
tooltip.style.display="none"
})

})