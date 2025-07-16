import tomllib
from pathlib import Path
from typing import Final, List

import torch
from pydantic_settings import BaseSettings

# TODO: Load from config with specific deployment.
# with open("configs/development.toml", "r", encoding="utf-8") as f:
#     CONFIG: Final = tomllib.load(f)

class Settings(BaseSettings):
    PROJECT_NAME: str = "Active Learning Medical Segmentation Web Server"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ]

    # Model settings
    DEFAULT_FOUNDATION_MODEL: str = "BiomedCLIP"
    DEFAULT_SPECIALIST_MODEL: str = "./data/models/init_model.pth"
    IMAGE_SIZE: int = 256
    IMAGES_PER_ROW: int = 10

    # Data directories
    ROOT_DIR: Path = Path("./resources")
    DATA_DIR: Path = ROOT_DIR / "data"
    MODELS_DIR: Path = ROOT_DIR / "models"
    DATASETS_DIR: Path = DATA_DIR / "datasets"

    # Device settings
    DEVICE: str = "cpu"

    # Active learning settings
    DEFAULT_BUDGET: int = 10
    DEFAULT_BATCH_SIZE: int = 4
    DEFAULT_LOADED_FEATURE_WEIGHT: float = 1.0
    DEFAULT_SHARP_FACTOR: float = 1.0
    DEFAULT_LOADED_FEATURE_ONLY: bool = False

    # File upload settings
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if torch.cuda.is_available() and self.DEVICE == "cpu":
            self.DEVICE = "cuda"


        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.DATASETS_DIR.mkdir(exist_ok=True)


class ActiveLearningConfig:
    def __init__(self):
        self.budget = settings.DEFAULT_BUDGET
        self.model = settings.DEFAULT_FOUNDATION_MODEL
        self.device = torch.device(settings.DEVICE)
        self.batch_size = settings.DEFAULT_BATCH_SIZE
        self.loaded_feature_weight = settings.DEFAULT_LOADED_FEATURE_WEIGHT
        self.sharp_factor = settings.DEFAULT_SHARP_FACTOR
        self.loaded_feature_only = settings.DEFAULT_LOADED_FEATURE_ONLY
        self.model_ckpt = settings.DEFAULT_SPECIALIST_MODEL


settings = Settings()
