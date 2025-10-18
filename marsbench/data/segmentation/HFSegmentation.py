"""
PyTorch Dataset class for loading segmentation datasets from Hugging Face Hub.
"""

import logging
from typing import List, Tuple, Optional, Callable, Literal

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image

from .BaseSegmentationDataset import BaseSegmentationDataset

logger = logging.getLogger(__name__)


class HFSegmentation(BaseSegmentationDataset):
    def __init__(
        self,
        cfg: DictConfig,
        repo_id: str,
        transform: Optional[Callable] = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.repo_id = repo_id
        self.hf_cache_dir = cfg.data.get("hf_cache_dir")
        self.images = []  
        self.masks = []  

        logger.info(f"Initializing HFSegmentation: repo='{self.repo_id}', split='{split}'")
        if self.hf_cache_dir:
            logger.info(f"Using Hugging Face cache directory: {self.hf_cache_dir}")

        super().__init__(cfg=cfg, data_dir=None, transform=transform, split=split)

    def _load_data(self) -> Tuple[List[str], List[str]]:
        logger.info(f"Loading Hugging Face dataset '{self.repo_id}' split '{self.split}'...")
        
        ds = load_dataset(
            self.repo_id,
            split=self.split,
            cache_dir=self.hf_cache_dir
        )

        image_column = "image"
        mask_column = "mask"

        if image_column not in ds.column_names or mask_column not in ds.column_names:
            raise ValueError(f"Dataset '{self.repo_id}' is missing required columns '{image_column}' or '{mask_column}'. Available columns: {ds.column_names}")

        image_paths = []
        mask_paths = []

        for idx, item in enumerate(ds):
            image_pil = item[image_column]
            mask_pil = item[mask_column]

            self.images.append(image_pil)
            self.masks.append(mask_pil)
            image_paths.append(f"hf_image_{idx}")  
            mask_paths.append(f"hf_mask_{idx}")   

        logger.info(f"Successfully extracted {len(image_paths)} images and masks from Hugging Face dataset.")
        return image_paths, mask_paths

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.array(self.images[idx].convert(self.image_type))
        mask = np.array(self.masks[idx].convert("L"))

        if len(mask.shape) == 4:  # One hot encoded mask [N, C, H, W] to class mask [N, H, W]
            mask = np.argmax(mask, axis=2)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

        mask = mask.to(torch.int64)
        return image, mask