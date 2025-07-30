from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from entry.demo.web.api.routes import api_router
from entry.demo.web.config import settings
from entry.demo.web.services.active_learning import active_learning_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting up FastAPI application...")
    print("Working in simplified mode (no heavy model dependencies)")
    await active_learning_service.specialist_model.initialize()
    yield

    # Shutdown
    print("Shutting down FastAPI application...")
    print("Application shutdown complete")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Active Learning API for Medical Image Segmentation (Simplified Mode)",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path("data").exists():
    app.mount("/static/data", StaticFiles(directory="data"), name="data")

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    return {
        "message": "Active Learning API for Medical Image Segmentation",
        "version": settings.VERSION,
        "mode": "simplified",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "simplified",
        "message": "API is running in simplified mode without heavy dependencies"
    }


@app.get("/api/info")
async def api_info():
    """Get API information and available endpoints."""
    return {
        "title": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "endpoints": {
            "active_learning": {
                "select": "POST /api/v1/active-learning/select",
                "config": "GET/POST /api/v1/active-learning/config",
                "predict": "POST /api/v1/active-learning/predict"
            },
            "models": {
                "status": "GET /api/v1/models/status",
                "list": "GET /api/v1/models/checkpoints",
                "upload": "POST /api/v1/models/checkpoints",
                "upload_file": "POST /api/v1/models/checkpoints/upload-file"
            },
            "datasets": {
                "upload_image": "POST /api/v1/datasets/images",
                "upload_file": "POST /api/v1/datasets/images/upload-file",
                "list_images": "GET /api/v1/datasets/images",
                "create_dataset": "POST /api/v1/datasets/datasets/{name}",
                "export": "POST /api/v1/datasets/export"
            }
        }
    }


def main():
    uvicorn.run(
        "entry.demo.web.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    main()
