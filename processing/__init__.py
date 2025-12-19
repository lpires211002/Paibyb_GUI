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

from .extraccion import (
    # Detectores de características
    apply_harris,
    apply_sift,
    apply_orb,
    
    # Métricas geométricas
    calcular_metricas_contorno,
    extraer_contorno_principal,
    dibujar_contorno,
    
    # Funciones de área
    calcular_relacion_areas,
    contar_area,
    binarizar_imagen,
    
    # Función principal
    extraer_todas_caracteristicas
)

__all__ = [
    # Procesamiento
    "process_single_image",
    "process_nlm_pipeline",

    # Segmentación
    "load_segmentation_model",
    "segment_image",
    "dice_score",
    "iou_score",

    # Clasificación
    "load_classification_model",
    "classify_image",
    
    # Extracción de características - Detectores
    "apply_harris",
    "apply_sift",
    "apply_orb",
    
    # Extracción de características - Métricas
    "calcular_metricas_contorno",
    "extraer_contorno_principal",
    "dibujar_contorno",
    "calcular_relacion_areas",
    "contar_area",
    "binarizar_imagen",
    "extraer_todas_caracteristicas"
]