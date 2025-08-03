from fastapi import APIRouter
from fastapi.params import Depends

from ...middleware.workspace import get_workspace
from ...services.active_learning import active_learning_service
from ...services.dataset import dataset_service
from .active_learning import router as active_learning_router
from .dataset import router as dataset_router
from .manager import router as manager_router
from .models import router as models_router

api_router = APIRouter()

api_router.include_router(manager_router, prefix="/manager")

api_router.include_router(active_learning_router, prefix="/active-learning")
api_router.include_router(dataset_router, prefix="/dataset")
api_router.include_router(models_router, prefix="/models")

@api_router.post("/reset")
def reset(workspace_id: str = Depends(get_workspace)):
    """Reset the API."""
    with active_learning_service.workspace(workspace_id) as ws:
        with dataset_service.workspace(workspace_id) as ds:
            try:
                ws.clear()
                ds.clear()
            except Exception as e:
                print(f"Failed to reset API: {e}")
                raise
            return {"message": "System clear successfully"}


__all__ = ["api_router"]
