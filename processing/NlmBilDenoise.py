import cv2
import numpy as np
from processing.metrics import compute_metrics, compute_snr


# ============================================
# Normalización
# ============================================
def normalize_image(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


# ============================================
# Bilateral → CLAHE → NLM
# ============================================
def preprocess_mri(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # 1. Bilateral
    bilateral = cv2.bilateralFilter(img, d=4, sigmaColor=15, sigmaSpace=15)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(bilateral)

    # 3. Non-Local Means
    nlm = cv2.fastNlMeansDenoising(
        clahe_img,
        h=6,
        templateWindowSize=4,
        searchWindowSize=21
    )

    return nlm.astype(np.float32)


# ============================================
# Procesamiento de imagen individual
# ============================================
def process_nlm_pipeline(image_array):
    """
    Recibe una imagen 2D numpy uint8
    """

    if image_array is None:
        raise ValueError("Imagen inválida (None).")

    img_norm = normalize_image(image_array)

    original_for_metrics = img_norm.astype(np.float32)

    processed = preprocess_mri(img_norm)

    psnr_val, ssim_val, snr_processed = compute_metrics(
        original_for_metrics, processed
    )

    snr_original = compute_snr(original_for_metrics)

    return {
        "original": image_array,
        "processed": processed.astype(np.uint8),
        "psnr": psnr_val,
        "ssim": ssim_val,
        "snr_original": snr_original,
        "snr_processed": snr_processed
    }

