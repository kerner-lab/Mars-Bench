"""
PyTorch Dataset class for loading detection datasets from Hugging Face Hub.
"""

import logging
from typing import List, Tuple, Optional, Callable, Literal

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image

from .BaseDetectionDataset import BaseDetectionDataset

logger = logging.getLogger(__name__)


class HFDetection(BaseDetectionDataset):
    def __init__(
        self,
        cfg: DictConfig,
        repo_id: str,
        transform: Optional[Callable] = None,
        bbox_format: Literal["coco", "yolo", "pascal_voc"] = "yolo",
        split: Literal["train", "val", "test"] = "train",
    ):
        self.repo_id = repo_id
        self.hf_cache_dir = cfg.data.get("hf_cache_dir")
        self.images = []  

        logger.info(f"Initializing HFDetection: repo='{self.repo_id}', split='{split}', bbox_format='{bbox_format}'")
        if self.hf_cache_dir:
            logger.info(f"Using Hugging Face cache directory: {self.hf_cache_dir}")

        super().__init__(cfg=cfg, data_dir=None, transform=transform, bbox_format=bbox_format, split=split)

    def _load_data(self) -> Tuple[List[str], List[List[float]], List[List[int]], List[str]]:
        logger.info(f"Loading Hugging Face dataset '{self.repo_id}' split '{self.split}'...")
        
        ds = load_dataset(
            self.repo_id,
            split=self.split,
            cache_dir=self.hf_cache_dir
        )

        image_column = "image"
        
        annotation_column_map = {
            "yolo": "yolo_annotation",
            "coco": "coco_annotation", 
            "pascal_voc": "pascal_voc_annotation"
        }
        
        annotation_column = annotation_column_map.get(self.bbox_format)
        if annotation_column is None:
            raise ValueError(f"Unsupported bbox_format: {self.bbox_format}. Supported formats: {list(annotation_column_map.keys())}")

        required_columns = [image_column, annotation_column]
        missing_columns = [col for col in required_columns if col not in ds.column_names]
        if missing_columns:
            raise ValueError(f"Dataset '{self.repo_id}' is missing required columns: {missing_columns}. Available: {ds.column_names}")

        image_paths = []
        annotations = []
        labels = []
        image_ids = []
        
        
        class_to_label = {}

        for idx, item in enumerate(ds):
            image_pil = item[image_column]
            annotation_data = item[annotation_column]

            # Store PIL image separately
            self.images.append(image_pil)
            image_paths.append(f"hf_image_{idx}")  
            
            # Parse annotations based on format
            item_bboxes, item_labels = self._parse_annotations(annotation_data, self.bbox_format, class_to_label)
            annotations.append(item_bboxes)
            labels.append(item_labels)
            image_ids.append(f"hf_id_{idx}")

       
        self.class_to_label = class_to_label
        logger.info(f"Class mapping: {class_to_label}")
        logger.info(f"Successfully extracted {len(image_paths)} images with annotations from Hugging Face dataset.")
        return image_paths, annotations, labels, image_ids

    def _parse_annotations(self, annotation_data, bbox_format, class_to_label):
        """Parse annotation data based on the bbox format."""
        bboxes = []
        labels = []
        
        if not annotation_data:  
            return bboxes, labels
            
        if bbox_format == "yolo":
            # YOLO format structure: {"bbox": [...], "category": [...]}
            bbox_list = annotation_data.get("bbox", [])
            category_list = annotation_data.get("category", [])
            
            for bbox, category in zip(bbox_list, category_list):
                bboxes.append(bbox)  # [x_center, y_center, width, height]
               
                if category not in class_to_label:
                    class_to_label[category] = len(class_to_label) + 1
                label = class_to_label[category]
                labels.append(label)
                    
        elif bbox_format == "coco":
            # COCO format structure: {"annotations": {"bbox": [...], "category_id": [...]}}
            annotations = annotation_data.get("annotations", {})
            bbox_list = annotations.get("bbox", [])
            category_ids = annotations.get("category_id", [])
            
            for bbox, category_id in zip(bbox_list, category_ids):
                bboxes.append(bbox)  # [x, y, width, height]
                labels.append(category_id)  
                
        elif bbox_format == "pascal_voc":
            # Pascal VOC format structure: {"objects": {"bbox": [...], "name": [...]}}
            objects = annotation_data.get("objects", {})
            bbox_list = objects.get("bbox", [])
            names = objects.get("name", [])
            
            for bbox_dict, class_name in zip(bbox_list, names):
                # Convert bbox dict to list [xmin, ymin, xmax, ymax]
                bbox = [bbox_dict["xmin"], bbox_dict["ymin"], bbox_dict["xmax"], bbox_dict["ymax"]]
                bboxes.append(bbox)
                
                if class_name not in class_to_label:
                    class_to_label[class_name] = len(class_to_label) + 1
                label = class_to_label[class_name]
                labels.append(label)
                    
        return bboxes, labels
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        image = self.images[idx].convert(self.image_type)
        bboxes = self.annotations[idx]
        labels = self.labels[idx]

        img_width, img_height = image.size
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed["image"]
            img_height, img_width = image.shape[-2:]
            bboxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "img_size": torch.tensor([img_height, img_width]),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target