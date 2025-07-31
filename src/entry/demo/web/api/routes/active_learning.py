import base64
import os
import json
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, Form
from fastapi.params import Query
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_404_NOT_FOUND

from entry.demo.web.models.requests import (ActiveLearningConfigRequest)
from entry.demo.web.models.responses import (ActiveLearningConfigResponse, ActiveLearningStateResponse)
from entry.demo.web.services.active_learning import active_learning_service
from entry.demo.web.services.dataset import dataset_service
from utils import draw_mask
from utils.images import class_color_map, hex_to_rgb, image_to_base64, base64_to_image


Logger = getLogger(__name__)

router = APIRouter()


@router.post("/config", response_model=ActiveLearningConfigResponse)
async def update_config(config: ActiveLearningConfigRequest):
    """Update active learning configuration."""
    return await active_learning_service.update_config(config)


@router.get("/state", response_model=ActiveLearningStateResponse)
async def get_state():
    """Get the current active learning state."""
    return active_learning_service.get_state()


@router.get("/config")
async def get_config():
    """Get the current active learning configuration."""
    return await active_learning_service.get_config()


@router.get("/selected-samples")
async def get_selected_samples():
    selected_images = []
    selected_set = active_learning_service.selected_set
    for path in selected_set:
        if os.path.exists(path):
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
                selected_images.append({
                    "path": path,
                    "name": os.path.basename(path),
                    "data": image_data
                })
    return {
        "message": "Get selected samples completed successfully",
        "selected_samples": selected_images,
        "count": len(selected_images)
    }


@router.post("/select-samples")
async def select_samples():
    train_set = active_learning_service.get_train_set()
    pool_set = active_learning_service.get_pool_set()
    annotated_set = active_learning_service.get_annotated_set()
    config = active_learning_service.config

    if not train_set and not annotated_set:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No dataset is loaded")

    if not pool_set:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No pool set is available")

    annotated_samples = [str(x["path"]) for x in active_learning_service.get_annotated_set()]
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
        for path in result:
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
            "count": len(selected_images)
        }

    except Exception as e:
        Logger.error(f"Error occurred during selection: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/annotated")
async def get_annotated_samples():
    """
    Get annotated samples with proper serialization
    """
    try:
        annotated_set = active_learning_service.get_annotated_set()

        serializable_samples = []
        for sample in annotated_set:
            visual_data = sample.get("visual", None)
            if visual_data is None:
                continue
            visual_base64 = image_to_base64(visual_data)
            serializable_samples.append(visual_base64)

        return {"annotated_samples": serializable_samples}

    except Exception as e:
        Logger.error(f"Error in get_annotated_samples: {e}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error retrieving annotated samples: {str(e)}")


@router.post("/annotate")
async def annotate_image(
    image_path: str = Form(...),
    background: str = Form(...),
    layers: Union[str, List[str]] = Form(...),
):
    """
    Annotate image endpoint
    """
    if image_path not in active_learning_service.selected_set:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Image not found")

    image_index = active_learning_service.selected_set.index(image_path)
    active_learning_service.selected_image = {
        "index": image_index,
        "case_name": Path(image_path).stem,
    }

    try:
        if isinstance(layers, str):
            try:
                layers_list = json.loads(layers)
                if not isinstance(layers_list, list):
                    layers_list = [layers_list]
            except json.JSONDecodeError:
                layers_list = [layers]
        elif isinstance(layers, list):
            layers_list = layers
        else:
            layers_list = [str(layers)]

        if not layers_list or len(layers_list) == 0:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No annotation layers provided")

        layer_base64 = layers_list[0]

        if layer_base64 is None or layer_base64 == 'null' or str(layer_base64).strip() in ['null', '[null]', 'None']:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Client sent null/invalid layer data. Received: '{layer_base64}'"
            )

        if not isinstance(layer_base64, str) or len(layer_base64) < 10:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Invalid layer data: {type(layer_base64)}, length: {len(layer_base64) if isinstance(layer_base64, str) else 'N/A'}"
            )

        try:
            image_pil = base64_to_image(background).convert("RGB")
            active_learning_service.selected_image["image"] = image_pil
        except Exception as e:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST,
                                detail=f"Failed to process background image: {str(e)}")

        try:
            layer_image = base64_to_image(layer_base64)
            layer_np = np.array(layer_image.convert("RGBA"))
        except Exception as e:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"Failed to process layer image: {str(e)}")

        binary_layer_np = np.zeros_like(layer_np)
        binary_layer_np[layer_np > 127] = 255

        H, W, _ = layer_np.shape
        mask_np = np.zeros((H, W), dtype=np.uint8)

        for cl, color_hex in class_color_map.items():
            try:
                color_rgb = hex_to_rgb(color_hex)
                bin_mask = np.all(binary_layer_np[:, :, :3] == color_rgb, axis=-1)
                mask_np[bin_mask] = cl

            except Exception as e:
                Logger.error(f"Error processing color {color_hex} for class {cl}: {e}")
                continue

        active_learning_service.selected_image["mask"] = mask_np
        visual_image = draw_mask(image_pil, mask_np)
        active_learning_service.selected_image["visual"] = visual_image

        annotated_path = active_learning_service.selected_set[image_index]

        # Remove annotated images
        os.remove(annotated_path)
        active_learning_service.selected_set.remove(annotated_path)
        active_learning_service.current_pool_set.remove(annotated_path)

        new_path = dataset_service.save_annotated_image(active_learning_service.selected_image)
        active_learning_service.selected_image["path"] = new_path
        active_learning_service.annotated_set.append(deepcopy(active_learning_service.selected_image))
        active_learning_service.update_feature_dict_keys(annotated_path, new_path)
        active_learning_service.selected_image = None

        visual_base64 = image_to_base64(visual_image)

        response_data = {
            "message": "Annotation accepted successfully",
            "annotated_count": len(active_learning_service.annotated_set),
            "remaining_count": len(active_learning_service.selected_set),
            "processed_classes": [int(x) for x in np.unique(mask_np) if x > 0],
            "visual_base64": visual_base64
        }
        return response_data

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        Logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"Invalid JSON in layers data: {str(e)}")
    except Exception as e:
        import traceback
        Logger.debug(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing error: {str(e)}")


@router.get("/pseudo-label")
async def get_pseudo_label(
    image_path: str = Query(..., description="Path to the image to be annotated"),
):
    if image_path not in active_learning_service.selected_set:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Image not found")

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