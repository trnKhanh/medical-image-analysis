from contextlib import asynccontextmanager
from pathlib import Path

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from entry.demo.web.api.routes import api_router
from entry.demo.web.config import settings
from entry.demo.web.services.active_learning import active_learning_service

Logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    Logger.info("Starting up FastAPI application...")
    await active_learning_service.specialist_model.initialize()
    yield

    Logger.info("Shutting down FastAPI application...")
    Logger.info("Application shutdown complete")


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
