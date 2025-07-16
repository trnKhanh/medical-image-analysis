import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Tuple

import torch
from open_clip import create_model_from_pretrained, get_tokenizer

from entry.demo.web.config import settings


class FoundationModelManager:
    def __init__(self):
        self.model: Optional[Any] = None
        self.preprocess: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device = torch.device(settings.DEVICE)
        self.is_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def initialize(self):
        """Initialize the foundation model."""
        if self.is_initialized:
            return

        def _load_model():
            if settings.DEFAULT_FOUNDATION_MODEL == "BiomedCLIP":
                model, preprocess = create_model_from_pretrained(
                    "hf-hub:microsoft/biomedclip-pubmedbert_256-vit_base_patch16_224"
                )
                tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
                model.to(self.device)
                model.eval()
                return model, preprocess, tokenizer
            else:
                raise RuntimeError(f"Unsupported foundation model: {settings.DEFAULT_FOUNDATION_MODEL}")

        try:
            loop = asyncio.get_event_loop()
            self.model, self.preprocess, self.tokenizer = await loop.run_in_executor(
                self.executor, _load_model
            )
            self.is_initialized = True
            print(f"Foundation model {settings.DEFAULT_FOUNDATION_MODEL} loaded successfully")
        except Exception as e:
            print(f"Failed to load foundation model: {e}")
            raise

    async def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.is_initialized and self.model is not None

    def get_model_and_preprocess(self) -> Tuple[Any, Any]:
        """Get the model and preprocessing function."""
        if not self.is_initialized:
            raise RuntimeError("Foundation model not initialized")
        return self.model, self.preprocess

    def get_tokenizer(self) -> Any:
        """Get the tokenizer."""
        if not self.is_initialized:
            raise RuntimeError("Foundation model not initialized")
        return self.tokenizer

    async def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.preprocess is not None:
            del self.preprocess
            self.preprocess = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_initialized = False
