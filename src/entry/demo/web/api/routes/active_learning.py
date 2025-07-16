from fastapi import APIRouter

from entry.demo.web.models.requests import (ActiveLearningConfigRequest,
                                            ActiveSelectionRequest,
                                            PredictionRequest)
from entry.demo.web.models.responses import (ActiveLearningConfigResponse,
                                             ActiveSelectionResponse,
                                             PredictionResponse)
from entry.demo.web.services.active_learning import active_learning_service

router = APIRouter()

@router.post("/select", response_model=ActiveSelectionResponse)
async def select_next_batch(request: ActiveSelectionRequest):
    """Perform active selection to get the next batch of images for annotation."""
    return await active_learning_service.select_next_batch_simple(request)

@router.post("/config", response_model=ActiveLearningConfigResponse)
async def update_config(config: ActiveLearningConfigRequest):
    """Update active learning configuration."""
    return await active_learning_service.update_config(config)

@router.get("/config", response_model=ActiveLearningConfigResponse)
async def get_config():
    """Get the current active learning configuration."""
    return await active_learning_service.get_config()

@router.post("/reset-features")
async def reset_feature_cache():
    """Reset the feature cache."""
    active_learning_service.reset_feature_cache()
    return {"message": "Feature cache reset successfully"}

@router.get("/session-data")
async def get_session_data():
    """Get current session data."""
    return active_learning_service.get_current_session_data()

@router.post("/clear-session")
async def clear_session():
    """Clear current session data."""
    active_learning_service.clear_session_data()
    return {"message": "Session data cleared successfully"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(request: PredictionRequest):
    """Generate a prediction for an image using a specialist model."""
    return await active_learning_service.predict_pseudo_label_simple(request)