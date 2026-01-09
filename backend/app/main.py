from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from app.config import settings
from app.database import init_db, close_db
from app.routers import datasets, networks, training, optimization


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    await init_db()
    os.makedirs(settings.upload_dir, exist_ok=True)
    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title=settings.app_name,
    description="Neural Network Optimization API with PSO and Genetic Algorithms",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for uploads
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

# Include routers
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(networks.router, prefix="/api/networks", tags=["Networks"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to NN Optimizer API",
        "docs": "/docs",
        "health": "/api/health",
    }
