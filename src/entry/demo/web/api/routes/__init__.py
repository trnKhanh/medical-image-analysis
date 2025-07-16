from fastapi import APIRouter

from .active_learning import router as active_learning_router
from .dataset import router as dataset_router
from .models import router as models_router

api_router = APIRouter()
api_router.include_router(active_learning_router, prefix="/active-learning")
api_router.include_router(dataset_router, prefix="/dataset")
api_router.include_router(models_router, prefix="/models")

__all__ = ["api_router"]
