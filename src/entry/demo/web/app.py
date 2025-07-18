# import base64
# import io
# import shutil
# import uvicorn
# import uuid
# import zipfile
# from pathlib import Path
# from typing import List, Optional
# import logging
#
# import numpy as np
# import torch
# import torchvision.transforms.functional as F
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from open_clip import create_model_from_pretrained, get_tokenizer
# from pydantic import BaseModel
# from torch.utils.data import ConcatDataset, DataLoader
#
# from activelearning import KMeanSelector
# from datasets import ActiveDataset, ExtendableDataset, ImageDataset
# from models.unet import UNet, UnetProcessor
# from utils import draw_mask
#
# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Constants
# IMAGES_PER_ROW = 10
# IMAGE_SIZE = 256
# ROOT_DIR = Path(".")
# DATA_DIR = ROOT_DIR / "data"
#
# # Ensure data directory exists
# DATA_DIR.mkdir(exist_ok=True)
#
# # Global variables
# train_set = []
# pool_set = []
# current_dataset = "dataset"
# feature_dict = None
# selected_image = None
# selected_set = []
# annotated_set = []
#
#
# class Config:
#     def __init__(self):
#         self.budget = 10
#         self.model = "BiomedCLIP"
#         self.device = torch.device("cpu")
#         self.batch_size = 4
#         self.loaded_feature_weight = 1.0
#         self.sharp_factor = 1.0
#         self.loaded_feature_only = False
#         self.model_ckpt = "./init_model.pth"
#
#
# config = Config()
#
# class_color_map = {
#     1: "#ff0000",
#     2: "#00ff00",
#     3: "#0000ff",
# }
#
#
# def build_foundation_model(device):
#     """Build and return foundation model with preprocessing"""
#     try:
#         if config.model == "BiomedCLIP":
#             model, preprocess = create_model_from_pretrained(
#                 "hf-hub:microsoft/biomedclip-pubmedbert_256-vit_base_patch16_224"
#             )
#             tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
#             model.to(device)
#             model.eval()
#             return model, preprocess
#         else:
#             raise RuntimeError("Unsupported model")
#     except Exception as e:
#         logger.error(f"Error building foundation model: {str(e)}")
#         raise
#
#
# def build_specialist_model():
#     """Build and return specialist UNet model"""
#     try:
#         model = UNet(
#             dimension=2,
#             input_channels=1,
#             output_classes=3,
#             channels_list=[32, 64, 128, 256, 512],
#             block_type="plain",
#             normalization="batch",
#         )
#         model_processor = UnetProcessor(image_size=(IMAGE_SIZE, IMAGE_SIZE))
#         return model, model_processor
#     except Exception as e:
#         logger.error(f"Error building specialist model: {str(e)}")
#         raise
#
#
# specialist_model, specialist_processor = build_specialist_model()
#
#
# def load_specialist_model(model_ckpt):
#     """Load specialist model weights from checkpoint"""
#     try:
#         if not Path(model_ckpt).exists():
#             logger.warning(f"Model checkpoint not found: {model_ckpt}")
#             return False
#
#         specialist_model.load_state_dict(
#             torch.load(model_ckpt, map_location=torch.device("cpu"), weights_only=True)
#         )
#         logger.info(f"Successfully loaded model checkpoint: {model_ckpt}")
#         return True
#     except Exception as e:
#         logger.error(f"Error loading model checkpoint {model_ckpt}: {str(e)}")
#         return False
#
#
# # Pydantic models
# class ConfigUpdate(BaseModel):
#     budget: Optional[int] = None
#     model_ckpt: Optional[str] = None
#     device: Optional[str] = None
#     batch_size: Optional[int] = None
#     loaded_feature_weight: Optional[float] = None
#     sharp_factor: Optional[float] = None
#     loaded_feature_only: Optional[bool] = None
#
#
# class ImageData(BaseModel):
#     path: str
#     image: Optional[str] = None
#
#
# class AnnotationData(BaseModel):
#     image_path: str
#     mask_data: List[List[int]]
#     background_image: str
#
#
# class SelectionResponse(BaseModel):
#     selected_images: List[str]
#     dataset_id: str
#
#
# class AnnotatedSample(BaseModel):
#     path: str
#     visual: str
#
#
# class CheckpointValidationRequest(BaseModel):
#     checkpoint_path: str
#
#
# # FastAPI app setup
# app = FastAPI(title="Active Learning Image Annotation API", version="1.0.0")
#
# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, specify your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # Utility functions
# def get_feature_dict(batch_size, device, active_dataset: ActiveDataset):
#     """Extract features for all images using foundation model"""
#     try:
#         dataset = ConcatDataset([active_dataset.get_train_dataset(), active_dataset.get_pool_dataset()])
#         dataloader = DataLoader(dataset, batch_size=batch_size)
#
#         model, preprocess = build_foundation_model(device)
#         feature_dict = {}
#
#         for sampled_batch in dataloader:
#             image_batch = sampled_batch["image"]
#             image_list = []
#             for image in image_batch:
#                 image_pil = F.to_pil_image(image).convert("RGB")
#                 image_list.append(preprocess(image_pil))
#             image_batch = torch.stack(image_list, dim=0)
#             image_batch = image_batch.to(device)
#
#             with torch.no_grad():
#                 feature_batch = model.encode_image(image_batch)
#
#             for i in range(len(feature_batch)):
#                 case_name = sampled_batch["case_name"][i]
#                 feature_dict[case_name] = feature_batch[i]
#
#         return feature_dict
#     except Exception as e:
#         logger.error(f"Error extracting features: {str(e)}")
#         raise
#
#
# def active_select(
#         train_set,
#         pool_set,
#         budget,
#         model_ckpt,
#         batch_size,
#         device,
#         loaded_feature_weight,
#         sharp_factor,
#         loaded_feature_only,
# ):
#     """Perform active learning sample selection"""
#     global feature_dict
#     try:
#         train_dataset = ExtendableDataset(ImageDataset(train_set, image_channels=1, image_size=IMAGE_SIZE))
#         pool_dataset = ExtendableDataset(ImageDataset(pool_set, image_channels=1, image_size=IMAGE_SIZE))
#         active_dataset = ActiveDataset(train_dataset, pool_dataset)
#
#         if feature_dict is None:
#             logger.info("Computing features for active learning...")
#             feature_dict = get_feature_dict(batch_size, device, active_dataset)
#
#         active_selector = KMeanSelector(
#             batch_size=4,
#             num_workers=1,
#             pin_memory=True,
#             metric="l2",
#             feature_dict=feature_dict,
#             loaded_feature_weight=loaded_feature_weight,
#             sharp_factor=sharp_factor,
#             loaded_feature_only=loaded_feature_only,
#         )
#
#         # Load model checkpoint
#         if not load_specialist_model(model_ckpt):
#             logger.warning("Using model without checkpoint")
#
#         return active_selector.select_next_batch(active_dataset, budget, specialist_model, device)
#     except Exception as e:
#         logger.error(f"Error in active selection: {str(e)}")
#         raise
#
#
# def predict_pseudo_label(image_pil):
#     """Generate pseudo labels for an image"""
#     try:
#         image = F.to_tensor(image_pil)
#         image = image.unsqueeze(0)
#         _, _, H, W = image.shape
#         image = specialist_processor.preprocess(image)
#
#         with torch.no_grad():
#             pred = specialist_model(image)
#             pseudo_label = pred.argmax(1)
#
#         pseudo_label = specialist_processor.postprocess(pseudo_label, [H, W])
#         return pseudo_label[0]
#     except Exception as e:
#         logger.error(f"Error predicting pseudo label: {str(e)}")
#         raise
#
#
# def hex_to_rgb(h):
#     """Convert hex color to RGB"""
#     h = h.lstrip('#')
#     return [int(h[i: i + 2], 16) for i in range(0, 6, 2)]
#
#
# def image_to_base64(image_pil):
#     """Convert PIL image to base64 string"""
#     try:
#         buffered = io.BytesIO()
#         image_pil.save(buffered, format="PNG")
#         return base64.b64encode(buffered.getvalue()).decode()
#     except Exception as e:
#         logger.error(f"Error converting image to base64: {str(e)}")
#         raise
#
#
# def base64_to_image(base64_str):
#     """Convert base64 string to PIL image"""
#     try:
#         image_data = base64.b64decode(base64_str)
#         return Image.open(io.BytesIO(image_data))
#     except Exception as e:
#         logger.error(f"Error converting base64 to image: {str(e)}")
#         raise
#
#
# def validate_image_file(file: UploadFile) -> bool:
#     """Validate if uploaded file is a valid image"""
#     valid_types = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"]
#     return file.content_type in valid_types
#
#
# # API endpoints
# @app.get("/")
# async def root():
#     return {
#         "message": "Active Learning Image Annotation API",
#         "version": "1.0.0",
#         "status": "running"
#     }
#
#
# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "train_images": len(train_set),
#         "pool_images": len(pool_set),
#         "selected_samples": len(selected_set),
#         "annotated_samples": len(annotated_set)
#     }
#
#
# @app.get("/config")
# async def get_config():
#     return {
#         "budget": config.budget,
#         "model_ckpt": config.model_ckpt,
#         "device": str(config.device),
#         "batch_size": config.batch_size,
#         "loaded_feature_weight": config.loaded_feature_weight,
#         "sharp_factor": config.sharp_factor,
#         "loaded_feature_only": config.loaded_feature_only
#     }
#
#
# @app.post("/config")
# async def update_config(config_update: ConfigUpdate):
#     global config, feature_dict
#
#     try:
#         updated_fields = []
#
#         if config_update.budget is not None:
#             config.budget = max(1, config_update.budget)  # Ensure positive budget
#             updated_fields.append("budget")
#
#         if config_update.model_ckpt is not None:
#             config.model_ckpt = config_update.model_ckpt
#             updated_fields.append("model_ckpt")
#
#         if config_update.device is not None:
#             try:
#                 config.device = torch.device(config_update.device)
#                 updated_fields.append("device")
#             except Exception:
#                 raise HTTPException(status_code=400, detail=f"Invalid device: {config_update.device}")
#
#         if config_update.batch_size is not None:
#             config.batch_size = max(1, config_update.batch_size)  # Ensure positive batch size
#             updated_fields.append("batch_size")
#
#         if config_update.loaded_feature_weight is not None:
#             config.loaded_feature_weight = max(0.0, config_update.loaded_feature_weight)
#             updated_fields.append("loaded_feature_weight")
#
#         if config_update.sharp_factor is not None:
#             config.sharp_factor = max(0.1, config_update.sharp_factor)
#             updated_fields.append("sharp_factor")
#
#         if config_update.loaded_feature_only is not None:
#             config.loaded_feature_only = config_update.loaded_feature_only
#             updated_fields.append("loaded_feature_only")
#
#         # Reset feature dict when significant config changes
#         if any(field in updated_fields for field in ["device", "batch_size"]):
#             feature_dict = None
#
#         logger.info(f"Configuration updated: {updated_fields}")
#         return {
#             "message": "Configuration updated successfully",
#             "updated_fields": updated_fields
#         }
#     except Exception as e:
#         logger.error(f"Error updating config: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#
#
# @app.post("/upload-train-images")
# async def upload_train_images(files: List[UploadFile] = File(...)):
#     global train_set, feature_dict
#
#     if not files:
#         raise HTTPException(status_code=400, detail="No files provided")
#
#     try:
#         train_set = []
#         upload_dir = DATA_DIR / "uploads" / "train"
#         upload_dir.mkdir(parents=True, exist_ok=True)
#
#         uploaded_count = 0
#         for file in files:
#             if not validate_image_file(file):
#                 logger.warning(f"Skipping invalid file: {file.filename}")
#                 continue
#
#             file_path = upload_dir / file.filename
#             with open(file_path, "wb") as buffer:
#                 content = await file.read()
#                 buffer.write(content)
#             train_set.append(str(file_path))
#             uploaded_count += 1
#
#         feature_dict = None  # Reset feature dict when train set changes
#         logger.info(f"Uploaded {uploaded_count} training images")
#         return {
#             "message": f"Uploaded {uploaded_count} training images",
#             "train_set_size": len(train_set)
#         }
#     except Exception as e:
#         logger.error(f"Error uploading training images: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.post("/upload-pool-images")
# async def upload_pool_images(files: List[UploadFile] = File(...)):
#     global pool_set, feature_dict
#
#     if not files:
#         raise HTTPException(status_code=400, detail="No files provided")
#
#     try:
#         pool_set = []
#         upload_dir = DATA_DIR / "uploads" / "pool"
#         upload_dir.mkdir(parents=True, exist_ok=True)
#
#         uploaded_count = 0
#         for file in files:
#             if not validate_image_file(file):
#                 logger.warning(f"Skipping invalid file: {file.filename}")
#                 continue
#
#             file_path = upload_dir / file.filename
#             with open(file_path, "wb") as buffer:
#                 content = await file.read()
#                 buffer.write(content)
#             pool_set.append(str(file_path))
#             uploaded_count += 1
#
#         feature_dict = None  # Reset feature dict when pool set changes
#         logger.info(f"Uploaded {uploaded_count} pool images")
#         return {
#             "message": f"Uploaded {uploaded_count} pool images",
#             "pool_set_size": len(pool_set)
#         }
#     except Exception as e:
#         logger.error(f"Error uploading pool images: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.get("/train-set")
# async def get_train_set():
#     return {
#         "train_set": train_set,
#         "count": len(train_set)
#     }
#
#
# @app.get("/pool-set")
# async def get_pool_set():
#     return {
#         "pool_set": pool_set,
#         "count": len(pool_set)
#     }
#
#
# @app.post("/select-samples")
# async def select_samples() -> SelectionResponse:
#     global selected_set, current_dataset, annotated_set
#
#     if not train_set and not annotated_set:
#         raise HTTPException(status_code=400, detail="No training data available")
#     if not pool_set:
#         raise HTTPException(status_code=400, detail="No pool data available")
#
#     try:
#         annotated_samples = [x["path"] for x in annotated_set]
#         logger.info(f"Starting active selection with budget: {config.budget}")
#
#         selected_set = active_select(
#             list(set(train_set + annotated_samples)),
#             pool_set,
#             config.budget,
#             config.model_ckpt,
#             config.batch_size,
#             config.device,
#             config.loaded_feature_weight,
#             config.sharp_factor,
#             config.loaded_feature_only,
#         )
#
#         current_dataset = str(uuid.uuid4())
#         logger.info(f"Selected {len(selected_set)} samples for annotation")
#
#         return SelectionResponse(
#             selected_images=selected_set,
#             dataset_id=current_dataset
#         )
#     except Exception as e:
#         logger.error(f"Error in sample selection: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.get("/selected-samples")
# async def get_selected_samples():
#     return {
#         "selected_set": selected_set,
#         "count": len(selected_set)
#     }
#
#
# @app.get("/pseudo-label/{image_path:path}")
# async def get_pseudo_label(image_path: str):
#     try:
#         if not Path(image_path).exists():
#             raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
#
#         image_pil = Image.open(image_path).convert("L")
#         background = np.array(image_pil.convert("RGBA"))
#         pseudo_label = predict_pseudo_label(image_pil).cpu().numpy()
#
#         layers = []
#
#         # Create separate layer for each class
#         for cl, color in class_color_map.items():
#             layer = np.zeros_like(background)
#             bin_mask = pseudo_label == cl
#             if np.any(bin_mask):  # Only add layer if class is present
#                 color_rgb = hex_to_rgb(color)
#                 layer[bin_mask] = color_rgb + [128]  # Semi-transparent
#                 layers.append(layer.tolist())
#
#         logger.info(f"Generated pseudo label for {image_path} with {len(layers)} layers")
#         return {
#             "background": background.tolist(),
#             "layers": layers,
#             "image_path": image_path,
#             "classes_found": len(layers)
#         }
#     except Exception as e:
#         logger.error(f"Error processing image {image_path}: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
#
#
# @app.post("/annotate")
# async def annotate_image(annotation: AnnotationData):
#     global selected_set, annotated_set
#
#     try:
#         background_image = base64_to_image(annotation.background_image)
#         background_np = np.array(background_image)
#
#         mask_np = np.array(annotation.mask_data, dtype=np.uint8)
#
#         image_pil = Image.fromarray(background_np).convert("RGB")
#         visual_pil = draw_mask(image_pil, mask_np)
#         visual_b64 = image_to_base64(visual_pil)
#
#         annotated_sample = {
#             "path": annotation.image_path,
#             "image": background_np,
#             "mask": mask_np,
#             "visual": visual_b64
#         }
#
#         if annotation.image_path in selected_set:
#             selected_set.remove(annotation.image_path)
#         annotated_set.append(annotated_sample)
#
#         logger.info(f"Annotation accepted for {annotation.image_path}")
#         return {
#             "message": "Annotation accepted",
#             "annotated_count": len(annotated_set),
#             "selected_remaining": len(selected_set)
#         }
#     except Exception as e:
#         logger.error(f"Error processing annotation: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Error processing annotation: {str(e)}")
#
#
# @app.get("/annotated-samples")
# async def get_annotated_samples():
#     return {
#         "annotated_samples": [
#             {
#                 "path": sample["path"],
#                 "visual": sample["visual"]
#             }
#             for sample in annotated_set
#         ],
#         "count": len(annotated_set)
#     }
#
#
# @app.get("/download-dataset")
# async def download_dataset():
#     if not annotated_set:
#         raise HTTPException(status_code=400, detail="No annotated samples available")
#
#     try:
#         dataset_dir = DATA_DIR / "dataset"
#         if dataset_dir.exists():
#             shutil.rmtree(dataset_dir)
#         dataset_dir.mkdir(exist_ok=True, parents=True)
#
#         images_dir = dataset_dir / "images"
#         labels_dir = dataset_dir / "labels"
#         images_dir.mkdir(exist_ok=True, parents=True)
#         labels_dir.mkdir(exist_ok=True, parents=True)
#
#         zip_buffer = io.BytesIO()
#
#         with zipfile.ZipFile(zip_buffer, "w") as archive:
#             for sample in annotated_set:
#                 case_name = Path(sample["path"]).stem
#                 image_np = sample["image"]
#                 label_np = sample["mask"]
#
#                 image_pil = Image.fromarray(image_np)
#                 label_pil = Image.fromarray(label_np)
#
#                 image_path = images_dir / f"{case_name}.png"
#                 label_path = labels_dir / f"{case_name}.png"
#
#                 image_pil.save(image_path)
#                 label_pil.save(label_path)
#
#                 archive.write(image_path, arcname=f"images/{case_name}.png")
#                 archive.write(label_path, arcname=f"labels/{case_name}.png")
#
#         zip_buffer.seek(0)
#         logger.info(f"Dataset downloaded with {len(annotated_set)} samples")
#
#         return StreamingResponse(
#             io.BytesIO(zip_buffer.read()),
#             media_type="application/zip",
#             headers={"Content-Disposition": "attachment; filename=dataset.zip"}
#         )
#     except Exception as e:
#         logger.error(f"Error creating dataset: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.delete("/reset")
# async def reset_system():
#     global train_set, pool_set, selected_set, annotated_set, feature_dict, current_dataset
#
#     try:
#         train_set = []
#         pool_set = []
#         selected_set = []
#         annotated_set = []
#         feature_dict = None
#         current_dataset = "dataset"
#
#         logger.info("System reset successfully")
#         return {"message": "System reset successfully"}
#     except Exception as e:
#         logger.error(f"Error resetting system: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.get("/status")
# async def get_status():
#     return {
#         "train_set_size": len(train_set),
#         "pool_set_size": len(pool_set),
#         "selected_set_size": len(selected_set),
#         "annotated_set_size": len(annotated_set),
#         "current_dataset": current_dataset,
#         "feature_dict_loaded": feature_dict is not None,
#         "device": str(config.device),
#         "model_checkpoint": config.model_ckpt
#     }
#
#
# @app.get("/model-checkpoints")
# async def get_model_checkpoints():
#     """Get list of available model checkpoint files"""
#     try:
#         # Define directories to search for model files
#         checkpoint_dirs = [
#             Path("."),  # Current directory
#             Path("./models"),  # Models directory
#             Path("./checkpoints"),  # Checkpoints directory
#             Path("./weights"),  # Weights directory
#         ]
#
#         # Common model file extensions
#         model_extensions = {".pth", ".pt", ".ckpt", ".pkl", ".bin", ".safetensors"}
#
#         available_checkpoints = []
#
#         for checkpoint_dir in checkpoint_dirs:
#             if checkpoint_dir.exists():
#                 # Search for model files in the directory
#                 for file_path in checkpoint_dir.rglob("*"):
#                     if file_path.is_file() and file_path.suffix.lower() in model_extensions:
#                         try:
#                             # Use relative path from current directory
#                             relative_path = file_path.relative_to(Path("."))
#                             available_checkpoints.append({
#                                 "name": file_path.name,
#                                 "path": str(relative_path),
#                                 "size": file_path.stat().st_size,
#                                 "modified": file_path.stat().st_mtime
#                             })
#                         except Exception as e:
#                             logger.warning(f"Error processing file {file_path}: {str(e)}")
#                             continue
#
#         # Remove duplicates and sort by name
#         seen_paths = set()
#         unique_checkpoints = []
#         for checkpoint in available_checkpoints:
#             if checkpoint["path"] not in seen_paths:
#                 seen_paths.add(checkpoint["path"])
#                 unique_checkpoints.append(checkpoint)
#
#         unique_checkpoints.sort(key=lambda x: x["name"])
#
#         logger.info(f"Found {len(unique_checkpoints)} model checkpoints")
#         return {
#             "checkpoints": unique_checkpoints,
#             "count": len(unique_checkpoints)
#         }
#
#     except Exception as e:
#         logger.error(f"Error retrieving model checkpoints: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error retrieving model checkpoints: {str(e)}")
#
#
# @app.post("/validate-checkpoint")
# async def validate_checkpoint(request: CheckpointValidationRequest):
#     """Validate if a checkpoint file exists and is accessible"""
#     try:
#         file_path = Path(request.checkpoint_path)
#
#         if not file_path.exists():
#             return {"valid": False, "error": "File does not exist"}
#
#         if not file_path.is_file():
#             return {"valid": False, "error": "Path is not a file"}
#
#         # Check file size
#         if file_path.stat().st_size == 0:
#             return {"valid": False, "error": "File is empty"}
#
#         try:
#             # Try to load the checkpoint to verify it's valid
#             checkpoint = torch.load(file_path, map_location=torch.device("cpu"), weights_only=True)
#
#             # Basic validation - check if it's a state dict
#             if not isinstance(checkpoint, dict):
#                 return {"valid": False, "error": "Invalid checkpoint format"}
#
#             return {
#                 "valid": True,
#                 "message": "Checkpoint is valid",
#                 "keys_count": len(checkpoint.keys()) if isinstance(checkpoint, dict) else 0
#             }
#         except Exception as e:
#             return {"valid": False, "error": f"Invalid checkpoint file: {str(e)}"}
#
#     except Exception as e:
#         logger.error(f"Error validating checkpoint: {str(e)}")
#         return {"valid": False, "error": f"Error validating checkpoint: {str(e)}"}
#
#
# if __name__ == "__main__":
#     logger.info("Starting Active Learning Image Annotation API...")
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


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
