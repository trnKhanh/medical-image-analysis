import base64
import os
import json
from logging import getLogger

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, Form
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_404_NOT_FOUND

from entry.demo.web.models.requests import (ActiveLearningConfigRequest,
                                            ActiveSelectionRequest,
                                            PredictionRequest)
from entry.demo.web.models.responses import (ActiveLearningConfigResponse,
                                             ActiveSelectionResponse,
                                             PredictionResponse, ActiveLearningStateResponse)
from entry.demo.web.services.active_learning import active_learning_service
from utils import draw_mask
from utils.images import class_color_map, hex_to_rgb, image_to_base64, base64_to_image


Logger = getLogger(__name__)

router = APIRouter()

@router.post("/select", response_model=ActiveSelectionResponse)
async def select_next_batch(request: ActiveSelectionRequest):
    """Perform active selection to get the next batch of images for annotation."""
    return await active_learning_service.select_next_batch_simple(request)

@router.post("/config", response_model=ActiveLearningConfigResponse)
async def update_config(config: ActiveLearningConfigRequest):
    """Update active learning configuration."""
    return await active_learning_service.update_config(config)

@router.get("/state", response_model=ActiveLearningStateResponse)
async def get_state():
    """Get the current active learning state."""
    return active_learning_service.get_state()

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

@router.post("/select/samples")
async def select_samples():
    train_set = active_learning_service.get_train_set()
    pool_set = active_learning_service.get_pool_set()
    annotated_set = active_learning_service.get_annotated_set()
    config = active_learning_service.config

    if not train_set and not annotated_set:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No dataset is loaded")

    if not pool_set:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No pool set is available")

    annotated_samples = [x["path"] for x in active_learning_service.get_annotated_set()]

    try:
        result = await active_learning_service.active_select(
            list(set(train_set + annotated_samples)),
            pool_set,
            config.budget,
            config.model_ckpt,
            config.batch_size,
            config.device,
            config.loaded_feature_weight,
            config.sharp_factor,
            config.loaded_feature_only
        )

        if not result:
            Logger.warning("No samples were selected by the active learning algorithm")
            return {
                "message": "No samples selected - pool may be exhausted or criteria too restrictive",
                "selected_images": [],
                "count": 0
            }

        active_learning_service.selected_set = result

        selected_images = []
        for path in active_learning_service.selected_set:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                    selected_images.append({
                        "path": path,
                        "name": os.path.basename(path),
                        "data": image_data
                    })

        return {
            "message": "Selection completed successfully",
            "selected_images": selected_images,
            "count": len(active_learning_service.selected_set)
        }

    except Exception as e:
        Logger.error(f"Error occurred during selection: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/selected")
async def get_selected_samples():
    selected_images = []
    for path in active_learning_service.selected_set:
        if os.path.exists(path):
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
                selected_images.append({
                    "path": path,
                    "name": os.path.basename(path),
                    "data": image_data
                })

    return {"selected_images": selected_images}

@router.get("/annotated")
async def get_annotated_samples():
    return {"annotated_samples": [x["visual"] for x in active_learning_service.get_annotated_set()]}


@router.post("/annotate")
async def annotate_image(
    image_index: int = Form(...),
    background: str = Form(...),
    layers: str = Form(...),
):

    for i in active_learning_service.selected_set:
        print("==========", i)
    if image_index >= len(active_learning_service.selected_set):
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Image not found")

    layers_list = json.loads(layers)

    if len(layers_list) > 0:
        layer_image = base64_to_image(layers_list[0])
        layer_np = np.array(layer_image)

        binary_layer_np = np.zeros_like(layer_np)
        binary_layer_np[layer_np > 127] = 255
        H, W, _ = layer_np.shape
        mask_np = np.zeros((H, W), dtype=np.uint8)

        for cl, color in class_color_map.items():
            color_rgb = hex_to_rgb(color)
            bin_mask = np.all(binary_layer_np[:, :, :3] == color_rgb, axis=-1)
            mask_np[bin_mask] = cl

        background_image = base64_to_image(background)
        visual_image = draw_mask(background_image.convert("RGB"), mask_np)

        annotation = {
            "index": image_index,
            "path": active_learning_service.selected_set[image_index],
            "image": np.array(background_image),
            "mask": mask_np,
            "visual": image_to_base64(visual_image)
        }

        annotated_path = active_learning_service.selected_set[image_index]
        active_learning_service.selected_set = [x for x in active_learning_service.selected_set if x != annotated_path]
        active_learning_service.annotated_set.append(annotation)

        return {
            "message": "Annotation accepted successfully",
            "annotated_count": len(active_learning_service.annotated_set),
            "remaining_count": len(active_learning_service.selected_set)
        }
    else:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No annotation layers provided")


@router.get("/pseudo-label/{image_index}")
async def get_pseudo_label(
    image_index: int
):
    if image_index >= len(active_learning_service.selected_set):
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Image not found")

    image_path = active_learning_service.selected_set[image_index]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Image file not found")

    image_pil = Image.open(image_path).convert("L")
    background = np.array(image_pil.convert("RGBA"))
    pseudo_label = active_learning_service.predict_pseudo_label(image_pil).cpu().numpy()

    layer = np.zeros_like(background)
    for cl, color in class_color_map.items():
        bin_mask = pseudo_label == cl
        layer[bin_mask] = hex_to_rgb(color) + [255]

    background_b64 = image_to_base64(Image.fromarray(background))
    layer_b64 = image_to_base64(Image.fromarray(layer))

    return {
        "background": background_b64,
        "layers": [layer_b64],
        "composite": None
    }