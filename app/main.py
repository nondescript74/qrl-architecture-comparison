"""
main.py — QRL Architecture Comparison Demo
FastAPI entry point for Railway deployment.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.routers.comparison import router as comparison_router

app = FastAPI(
    title="QRL Architecture Comparison Demo",
    description="LLM-Only vs Decision Architecture — live side-by-side comparison",
    version="1.0.0",
)

# CORS — open for demo/slide embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount comparison router
app.include_router(comparison_router, prefix="/comparison", tags=["comparison"])

# Serve static frontend (the animation HTML)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    index = os.path.join(static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {
        "service": "QRL Architecture Comparison Demo",
        "endpoints": {
            "stream":  "/comparison/stream",
            "config":  "/comparison/config",
            "health":  "/comparison/health",
            "reset":   "/comparison/reset",
            "docs":    "/docs",
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
