import logging
from typing import List, Tuple, Optional, Callable, Literal


import numpy as np
import torch
from datasets import load_dataset 
from omegaconf import DictConfig
from PIL import Image
import ast


from .BaseClassificationDataset import BaseClassificationDataset


logger = logging.getLogger(__name__)


class HFClassification(BaseClassificationDataset):

    def __init__(
        self,
        cfg: DictConfig,
        repo_id: str,
        transform,
        split: Literal["train", "test", "val"]
    ):
        self.repo_id = repo_id
        self.split = split
        self.hf_cache_dir = cfg.data.get("hf_cache_dir")
        self.images = []  

        logger.info(f"Initializing HFClassification: repo='{self.repo_id}', split='{self.split}'")
        if self.hf_cache_dir:
            logger.info(f"Using Hugging Face cache directory: {self.hf_cache_dir}")

        super().__init__(cfg=cfg, data_dir=None, transform=transform)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        logger.info(f"Loading Hugging Face dataset '{self.repo_id}' split '{self.split}'...")
        
        ds = load_dataset(
            self.repo_id,
            split=self.split,
            cache_dir=self.hf_cache_dir
        )

        image_column = "image"
        label_column = "label"

        if image_column not in ds.column_names or label_column not in ds.column_names:
            raise ValueError(f"Dataset '{self.repo_id}' is missing required columns '{image_column}' or '{label_column}'. Available columns: {ds.column_names}")

        image_paths = []
        labels = []

        for idx, item in enumerate(ds):
            image_pil = item[image_column]
            label = item[label_column]

            if not isinstance(label, int):
                try:
                    label = int(label)
                except ValueError:
                    logger.error(f"Could not convert label '{label}' to int for item.")
                    raise 

            self.images.append(image_pil)
            image_paths.append(f"hf_image_{idx}")  
            labels.append(label)

        logger.info(f"Successfully extracted {len(image_paths)} images and labels from Hugging Face dataset.")
        return image_paths, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int | torch.Tensor]:
        
        image = np.array(self.images[idx].convert(self.image_type))
        
        if self.cfg.data.subtask == "multilabel":
            label = torch.zeros(self.cfg.data.num_classes, dtype=torch.float32)
            label[ast.literal_eval(str(self.gts[idx]))] = 1
        elif self.cfg.data.subtask == "binary":
            label = torch.tensor(self.gts[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.gts[idx], dtype=torch.long)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image)

        return image, label
