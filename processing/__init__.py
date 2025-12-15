from .GaussDenoise import process_single_image
from .NlmBilDenoise import process_nlm_pipeline

from .segmentador import (
    load_segmentation_model,
    segment_image,
    dice_score,
    iou_score
)

from .clasificacion import (
    load_classification_model,
    classify_image
)

__all__ = [
    "process_single_image",
    "process_nlm_pipeline",

    # Segmentación
    "load_segmentation_model",
    "segment_image",
    "dice_score",
    "iou_score",

    # Clasificación
    "load_classification_model",
    "classify_image"
]

