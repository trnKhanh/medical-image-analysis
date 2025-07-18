from fastapi import APIRouter

from .active_learning import router as active_learning_router
from .dataset import router as dataset_router
from .models import router as models_router
from ...services.active_learning import active_learning_service
from ...services.dataset import dataset_service

api_router = APIRouter()
api_router.include_router(active_learning_router, prefix="/active-learning")
api_router.include_router(dataset_router, prefix="/dataset")
api_router.include_router(models_router, prefix="/models")

@api_router.post("/reset")
def reset():
    """Reset the API."""
    dataset_service.clear()
    active_learning_service.clear()
    return {"message": "Dataset clear successfully"}


__all__ = ["api_router"]
