"""
Dataset loading and preprocessing utilities for MarsBench.
"""

import logging
import os
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data import Subset

from huggingface_hub import HfApi, HfFolder, login, whoami
from huggingface_hub.utils import HfHubHTTPError

from marsbench.data.segmentation.Mask2FormerWrapper import Mask2FormerWrapper

from .classification import Atmospheric_Dust_Classification_EDR
from .classification import Atmospheric_Dust_Classification_RDR
from .classification import Change_Classification_CTX
from .classification import Change_Classification_HiRISE
from .classification import DoMars16k
from .classification import Frost_Classification
from .classification import Landmark_Classification
from .classification import Multi_Label_MER
from .classification import Surface_Classification
from .classification import HFClassification
from .detection import Boulder_Detection
from .detection import ConeQuest_Detection
from .detection import Dust_Devil_Detection
from .detection import HFDetection
from .segmentation import MMLS
from .segmentation import Boulder_Segmentation
from .segmentation import ConeQuest_Segmentation
from .segmentation import Crater_Binary_Segmentation
from .segmentation import Crater_Multi_Segmentation
from .segmentation import MarsSegMER
from .segmentation import MarsSegMSL
from .segmentation import S5Mars
from .segmentation import HFSegmentation

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "classification": {
        "DoMars16k": DoMars16k,
        "Landmark_Classification": Landmark_Classification,
        "Surface_Classification": Surface_Classification,
        "Frost_Classification": Frost_Classification,
        "Atmospheric_Dust_Classification_RDR": Atmospheric_Dust_Classification_RDR,
        "Atmospheric_Dust_Classification_EDR": Atmospheric_Dust_Classification_EDR,
        "Change_Classification_HiRISE": Change_Classification_HiRISE,
        "Change_Classification_CTX": Change_Classification_CTX,
        "Multi_Label_MER": Multi_Label_MER,
    },
    "segmentation": {
        "ConeQuest_Segmentation": ConeQuest_Segmentation,
        "Boulder_Segmentation": Boulder_Segmentation,
        "MarsSegMER": MarsSegMER,
        "MarsSegMSL": MarsSegMSL,
        "MMLS": MMLS,
        "S5Mars": S5Mars,
        "Crater_Binary_Segmentation": Crater_Binary_Segmentation,
        "Crater_Multi_Segmentation": Crater_Multi_Segmentation,
    },
    "detection": {
        "ConeQuest_Detection": ConeQuest_Detection,
        "Dust_Devil_Detection": Dust_Devil_Detection,
        "Boulder_Detection": Boulder_Detection,
    },
}

HF_REGISTRY = {
    "classification": {
        "repo_ids": {
            "DoMars16k": "Mirali33/mb-domars16k",
            "Landmark_Classification": "Mirali33/mb-landmark_cls",
            "Surface_Classification": "Mirali33/mb-surface_cls",
            "Frost_Classification": "Mirali33/mb-frost_cls",
            "Atmospheric_Dust_Classification_RDR": "Mirali33/mb-atmospheric_dust_cls_rdr",
            "Atmospheric_Dust_Classification_EDR": "Mirali33/mb-atmospheric_dust_cls_edr",
            "Change_Classification_HiRISE": "Mirali33/mb-change_cls_hirise",
            "Change_Classification_CTX": "Mirali33/mb-change_cls_ctx",
            "Multi_Label_MER": "Mirali33/mb-surface_multi_label_cls",
        },
        "class": HFClassification,
    },
    "segmentation": {
        "repo_ids": {
            "ConeQuest_Segmentation": "Mirali33/mb-conequest_seg",
            "Boulder_Segmentation": "Mirali33/mb-boulder_seg",
            "MarsSegMER": "Mirali33/mb-mars_seg_mer",
            "MarsSegMSL": "Mirali33/mb-mars_seg_msl",
            "MMLS": "Mirali33/mb-mmls",
            "S5Mars": "Mirali33/mb-s5mars",
            "Crater_Binary_Segmentation": "Mirali33/mb-crater_binary_seg",
            "Crater_Multi_Segmentation": "Mirali33/mb-crater_multi_seg",
        },
        "class": HFSegmentation,
    },
    "detection": {
        "repo_ids": {
            "ConeQuest_Detection": "Mirali33/mb-conequest_det",
            "Dust_Devil_Detection": "Mirali33/mb-dust_devil_det",
            "Boulder_Detection": "Mirali33/mb-boulder_det",
        },
        "class": HFDetection,
    }
}


def instantiate_dataset(dataset_class, cfg, transform, split, bbox_format=None, annot_csv=None):
    common_args = {
        "cfg": cfg,
        "data_dir": cfg.data.data_dir,
        "transform": transform,
        "split": split,
    }
    partition_support = ["classification", "segmentation", "detection"]
    few_shot_support = ["classification"]
    if (cfg.partition is not None and cfg.task in partition_support) or (
        cfg.few_shot is not None and cfg.task in few_shot_support
    ):
        if cfg.few_shot is not None and cfg.partition is not None:
            msg = "At most one of cfg.few_shot or cfg.partition may be set; both cannot be non-None"
            logger.error(msg)
            raise ValueError(msg)
        if cfg.few_shot is not None:
            annot_csv = cfg.data.few_shot_csv
        elif cfg.partition is not None:
            annot_csv = cfg.data.partition_csv
            if cfg.partition >= 0.1:
                annot_csv_partition = annot_csv.split("/")[-1]
                annot_csv_partition_val = annot_csv_partition.split("x_")[0]
                new_val = f"{cfg.partition:.2f}"
                logger.info(f"Old annotation csv path is: {annot_csv}")
                annot_csv = annot_csv.replace(f"{annot_csv_partition_val}x_", f"{new_val}x_")
                logger.info(f"Updated annotation csv path is: {annot_csv}")
    elif cfg.partition is not None:
        logger.warning(f"Task: {cfg.task} does not support partition. Using whole data.")
    elif cfg.few_shot is not None:
        logger.warning(f"Task: {cfg.task} does not support few shot. Using whole data.")

    if annot_csv is not None and not os.path.exists(annot_csv):
        logger.error(f"Annotation csv path does not exist: {annot_csv}")
        raise ValueError(f"Annotation csv path does not exist: {annot_csv}")

    common_args["annot_csv"] = cfg.data.annot_csv if annot_csv is None and cfg.task != "detection" else annot_csv
    if cfg.task == "detection":
        common_args["bbox_format"] = bbox_format
    return dataset_class(**common_args)


def instantiate_dataset_hf(hf_class, cfg, transform, split, bbox_format=None, annot_csv=None):
    original_split = split
    partition_support = ["classification", "segmentation", "detection"]
    few_shot_support = ["classification"]
    
    if (cfg.partition is not None and cfg.task in partition_support) or (
        cfg.few_shot is not None and cfg.task in few_shot_support
    ):
        if cfg.few_shot is not None and cfg.partition is not None:
            msg = "At most one of cfg.few_shot or cfg.partition may be set; both cannot be non-None"
            logger.error(msg)
            raise ValueError(msg)
        
        if cfg.few_shot is not None and split == "train":
            split = f"few_shot_train_{cfg.few_shot}_shot"
            logger.info(f"Using few-shot split: {split}")
        elif cfg.partition is not None and split == "train":
            if cfg.task == "detection":
                split = f"{cfg.partition:.2f}x_partition"
            else:
                split = f"partition_train_{cfg.partition:.2f}x_partition"
            logger.info(f"Using partition split: {split}")
    elif cfg.partition is not None:
        logger.warning(f"Task: {cfg.task} does not support partition. Using whole data.")
    elif cfg.few_shot is not None:
        logger.warning(f"Task: {cfg.task} does not support few shot. Using whole data.")

    common_args = {
        "cfg": cfg,
        "repo_id": cfg.repo_id,
        "transform": transform,
        "split": split,
    }
    if cfg.task == "detection":
        common_args["bbox_format"] = bbox_format
    return hf_class(**common_args)


def is_hf_logged_in():
    token = HfFolder.get_token()
    if token is None:
        return False
    try:
        user_info = whoami(token)
        return True
    except HfHubHTTPError:
        return False



def get_dataset(
    cfg: DictConfig,
    transforms: Tuple[torch.nn.Module, torch.nn.Module],
    subset: Union[int, None] = None,
    bbox_format: Optional[str] = None,
    annot_csv: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Returns a train, val, and test dataset.

    Args:
        cfg (DictConfig):
            Configuration dictionary.
        transforms (Tuple[torch.nn.Module, torch.nn.Module]):
            Tuple of train and val transforms.
        subset (Union[int, None], optional):
            Number of samples to use for training. Prioritizes cfg.data.subset over this argument.
        annot_csv (Optional[str], optional):
            Path to annotation CSV file.
    Returns:
        Tuple[Dataset, Dataset, Dataset]: Tuple of train, val, and test datasets.
    """

    if not cfg.load_from_hf:
        try:
            dataset_cls = DATASET_REGISTRY[cfg.task][cfg.data.name]
        except KeyError as e:
            logger.error(f"Unsupported dataset {cfg.data.name} for task {cfg.task}")
            logger.debug(f"Available datasets for task {cfg.task}: {DATASET_REGISTRY[cfg.task]}")
            raise ValueError(f"Dataset not supported: {cfg.data.name} for {cfg.task}") from e

        train_dataset = instantiate_dataset(dataset_cls, cfg, transforms[0], "train", bbox_format, annot_csv)
        val_dataset = instantiate_dataset(dataset_cls, cfg, transforms[1], "val", bbox_format, annot_csv)
        test_dataset = instantiate_dataset(dataset_cls, cfg, transforms[1], "test", bbox_format, annot_csv)

    else:
        if not cfg.repo_id:
            logger.warning(f"Using Hugging Face without specifying a repository. Defaulting to {HF_REGISTRY[cfg.task]['repo_ids'][cfg.data.name]}")
            cfg.repo_id = HF_REGISTRY[cfg.task]["repo_ids"][cfg.data.name]
            logger.warning(f"Using default repository ID: {cfg.repo_id}")
        
        if not is_hf_logged_in():
            print("Hugging Face user not logged in. Please run `huggingface-cli login` or login below.")
            login() 
        else:
            logger.warning("Hugging Face user not logged in. Please run `huggingface-cli login` or login below.")
            login() 
        else:
            logger.info("Hugging Face user is logged in")

        # hf class
        hf_class = HF_REGISTRY[cfg.task]["class"]
        logger.info(f"Using Hugging Face dataset '{cfg.repo_id}' for task {cfg.task}")
        train_dataset = instantiate_dataset_hf(hf_class, cfg, transforms[0],
                                            "train", bbox_format, annot_csv)
        val_dataset   = instantiate_dataset_hf(hf_class, cfg, transforms[1],
                                            "val",   bbox_format, annot_csv)
        test_dataset  = instantiate_dataset_hf(hf_class, cfg, transforms[1],
                                            "test",  bbox_format, annot_csv)

    # Dataset wrapper for Mask2Former
    if cfg.model.name.lower() == "mask2former":
        train_dataset = Mask2FormerWrapper(train_dataset)
        val_dataset = Mask2FormerWrapper(val_dataset)
        test_dataset = Mask2FormerWrapper(test_dataset)

    # Apply subset if specified (prioritizing cfg.data.subset)
    actual_subset = cfg.data.get("subset", None) or subset
    if actual_subset is not None and actual_subset > 0:
        indices = torch.randperm(len(train_dataset))[:actual_subset].tolist()
        train_dataset = Subset(train_dataset, indices)

    return train_dataset, val_dataset, test_dataset
