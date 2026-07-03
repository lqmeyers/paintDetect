"""paintdetect: U-Net semantic segmentation of honeybee images.

Detects paint marks (individual bee ID) and body parts (head, thorax, abdomen)
via multi-class U-Net segmentation, built on a fork of milesial/Pytorch-UNet.
"""

from .model import UNet
from .data import LabelMeDataset, preprocess_image, assemble_mask_from_xml, get_xml_mask_dict
from .serialization import save_model, load_model
from .inference import predict_img, mask_to_image, run_test_eval
from .evaluate import evaluate, pixel_wise_confusion_matrix
from .metrics import calculate_iou, dice_coeff, multiclass_dice_coeff, dice_loss
from .config import load_config
from .train import train_model

__version__ = "0.1.0"

__all__ = [
    "UNet",
    "LabelMeDataset",
    "preprocess_image",
    "assemble_mask_from_xml",
    "get_xml_mask_dict",
    "save_model",
    "load_model",
    "predict_img",
    "mask_to_image",
    "run_test_eval",
    "evaluate",
    "pixel_wise_confusion_matrix",
    "calculate_iou",
    "dice_coeff",
    "multiclass_dice_coeff",
    "dice_loss",
    "load_config",
    "train_model",
    "__version__",
]
