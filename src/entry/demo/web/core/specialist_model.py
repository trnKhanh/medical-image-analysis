import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch

from entry.demo.web.config import settings
from models.unet import UNet, UnetProcessor


class SpecialistModelManager:
    def __init__(self):
        self.model: Optional[UNet] = None
        self.processor: Optional[UnetProcessor] = None
        self.current_checkpoint: Optional[str] = None
        self.device = torch.device(settings.DEVICE)
        self.is_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def initialize(self):
        """Initialize the specialist model architecture."""
        if self.is_initialized:
            return

        def _build_model():
            model = UNet(
                dimension=2,
                input_channels=1,
                output_classes=3,
                channels_list=[32, 64, 128, 256, 512],
                block_type="plain",
                normalization="batch",
            )
            processor = UnetProcessor(image_size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            return model, processor

        try:
            loop = asyncio.get_event_loop()
            self.model, self.processor = await loop.run_in_executor(
                self.executor, _build_model
            )
            self.is_initialized = True

            if Path(settings.DEFAULT_SPECIALIST_MODEL).exists():
                await self.load_model(settings.DEFAULT_SPECIALIST_MODEL)

            print("Specialist model architecture initialized successfully")
        except Exception as e:
            print(f"Failed to initialize specialist model: {e}")
            raise

    async def load_model(self, checkpoint_path: str):
        """Load a specific model checkpoint."""
        if not self.is_initialized:
            await self.initialize()

        def _load_checkpoint():
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            return checkpoint_path

        try:
            loop = asyncio.get_event_loop()
            self.current_checkpoint = await loop.run_in_executor(
                self.executor, _load_checkpoint
            )
            print(f"Specialist model checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load model checkpoint: {e}")
            raise

    async def is_loaded(self) -> bool:
        """Check if the model is loaded with a checkpoint."""
        return (self.is_initialized and
                self.model is not None and
                self.current_checkpoint is not None)

    def get_model(self) -> UNet:
        """Get the specialist model."""
        if not self.is_initialized or self.model is None:
            raise RuntimeError("Specialist model not initialized")
        return self.model

    def get_processor(self) -> UnetProcessor:
        """Get the model processor."""
        if not self.is_initialized or self.processor is None:
            raise RuntimeError("Specialist model processor not initialized")
        return self.processor

    async def validate_model(self, checkpoint_path: str) -> bool:
        """Validate a model checkpoint by attempting to load it."""
        try:
            def _validate():
                if not Path(checkpoint_path).exists():
                    return False

                state_dict = torch.load(
                    checkpoint_path,
                    map_location=torch.device("cpu"),
                    weights_only=True
                )

                temp_model = UNet(
                    dimension=2,
                    input_channels=1,
                    output_classes=3,
                    channels_list=[32, 64, 128, 256, 512],
                    block_type="plain",
                    normalization="batch",
                )

                temp_model.load_state_dict(state_dict)
                return True

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, _validate)
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.current_checkpoint = None
        self.is_initialized = False