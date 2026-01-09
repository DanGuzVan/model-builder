"""
NN Optimizer - FastAPI Main Application.

This is the main entry point for the Neural Network Optimization API.
It provides endpoints for dataset management, network architecture configuration,
training with monitoring, and hyperparameter optimization using PSO and GA.

Features:
- RESTful API with automatic OpenAPI documentation
- Background task processing for training and optimization
- CORS support for frontend integration
- Static file serving for uploads
- Health check endpoints
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import logging
import time

from app.config import settings
from app.database import init_db, close_db
from app.routers import datasets, networks, training, optimization

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events handler.
    
    Startup:
    - Initialize database tables
    - Create uploads directory
    
    Shutdown:
    - Close database connections
    """
    # Startup
    logger.info("Starting NN Optimizer API...")
    
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Ensure uploads directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info(f"Upload directory ready: {settings.upload_dir}")
    
    logger.info("NN Optimizer API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NN Optimizer API...")
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
## Neural Network Optimization API

A comprehensive API for training and optimizing neural networks using evolutionary algorithms.

### Features

- **Dataset Management**: Upload, preview, and manage CSV datasets
- **Network Architecture**: Define and configure neural network architectures
- **Training**: Train networks with real-time metrics and early stopping
- **Optimization**: Optimize hyperparameters using PSO and Genetic Algorithms

### Algorithms

- **PSO (Particle Swarm Optimization)**: Best for continuous parameters like learning rate
- **GA (Genetic Algorithm)**: Best for discrete architecture decisions

### Scientific Basis

This application implements research-backed optimization techniques:
- Early stopping (Prechelt, 1998)
- Learning rate as critical hyperparameter (Bengio, 2012)
- PSO for continuous optimization (Kennedy & Eberhart, 1995)
- GA for discrete architecture search (Holland, 1975)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# CORS Middleware Configuration
# Allow requests from frontend development server and production
cors_origins = settings.cors_origins + [
    "http://localhost:3000",      # React dev server
    "http://localhost:5173",      # Vite dev server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://frontend:80",         # Docker frontend service
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Request timing middleware (optional, useful for debugging)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to responses for performance monitoring."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions gracefully."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred",
            "error_type": type(exc).__name__,
        }
    )


# Mount static files for uploads
try:
    os.makedirs(settings.upload_dir, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
except Exception as e:
    logger.warning(f"Could not mount uploads directory: {e}")


# Include API routers with /api prefix
app.include_router(
    datasets.router, 
    prefix="/api/datasets", 
    tags=["Datasets"],
)
app.include_router(
    networks.router, 
    prefix="/api/networks", 
    tags=["Networks"],
)
app.include_router(
    training.router, 
    prefix="/api/training", 
    tags=["Training"],
)
app.include_router(
    optimization.router, 
    prefix="/api/optimization", 
    tags=["Optimization"],
)


# Health check and root endpoints
@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
    - status: "healthy" if API is running
    - app_name: Name of the application
    - version: API version
    - database: Database connection status
    - ollama: LLM service status (if available)
    """
    # Check database connectivity
    db_status = "unknown"
    try:
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check Ollama connectivity
    ollama_status = "unknown"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                ollama_status = "connected"
            else:
                ollama_status = f"error: HTTP {response.status_code}"
    except Exception as e:
        ollama_status = f"unavailable: {str(e)}"
    
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": "1.0.0",
        "database": db_status,
        "ollama": ollama_status,
        "debug_mode": settings.debug,
    }


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    
    Returns links to documentation and health check.
    """
    return {
        "message": f"Welcome to {settings.app_name} API",
        "version": "1.0.0",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "health": "/api/health",
            "datasets": "/api/datasets",
            "networks": "/api/networks",
            "training": "/api/training",
            "optimization": "/api/optimization",
        },
    }


@app.get("/api", tags=["Root"])
async def api_root():
    """
    API root endpoint.
    
    Returns summary of available endpoints.
    """
    return {
        "message": "NN Optimizer API",
        "version": "1.0.0",
        "endpoints": {
            "datasets": {
                "list": "GET /api/datasets",
                "upload": "POST /api/datasets/upload",
                "get": "GET /api/datasets/{id}",
                "preview": "GET /api/datasets/{id}/preview",
                "delete": "DELETE /api/datasets/{id}",
            },
            "networks": {
                "list": "GET /api/networks",
                "create": "POST /api/networks",
                "get": "GET /api/networks/{id}",
                "update": "PUT /api/networks/{id}",
                "delete": "DELETE /api/networks/{id}",
                "suggest": "POST /api/networks/suggest",
            },
            "training": {
                "list": "GET /api/training",
                "start": "POST /api/training",
                "get": "GET /api/training/{id}",
                "metrics": "GET /api/training/{id}/metrics",
                "cancel": "POST /api/training/{id}/cancel",
                "delete": "DELETE /api/training/{id}",
            },
            "optimization": {
                "list": "GET /api/optimization",
                "start": "POST /api/optimization",
                "get": "GET /api/optimization/{id}",
                "history": "GET /api/optimization/{id}/history",
                "cancel": "POST /api/optimization/{id}/cancel",
                "delete": "DELETE /api/optimization/{id}",
                "algorithms": "GET /api/optimization/algorithms/info",
            },
        },
    }


# Development server entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )