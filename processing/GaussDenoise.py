import cv2
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ============================================================
# SNR por bloques
# ============================================================
def compute_snr(image, block_size=20):
    image = image.astype(np.float64)
    h, w = image.shape

    snr_blocks = []

    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            block = image[r:min(r + block_size, h), c:min(c + block_size, w)]

            if block.size > 0:
                std_val = np.std(block)
                mean_val = np.mean(block)

                if std_val > 1e-6:
                    snr_blocks.append(20 * np.log10(mean_val / std_val))
                else:
                    snr_blocks.append(0.0)

    if not snr_blocks:
        return 0.0

    return np.mean(snr_blocks)


# ============================================================
# Métricas PSNR, SSIM y SNR
# ============================================================
def compute_metrics(original, processed):
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    psnr_val = psnr(original, processed, data_range=255)
    ssim_val = ssim(original, processed, data_range=255)
    snr_val = compute_snr(processed)

    return psnr_val, ssim_val, snr_val


# ============================================================
# Estimar ruido con bloques aleatorios
# ============================================================
def get_noise_stats_from_random_blocks(image, block_size=20, num_blocks=60):
    h, w = image.shape
    histograms = []
    block_means = []

    max_x = max(1, w - block_size)
    max_y = max(1, h - block_size)

    for _ in range(num_blocks):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        block = image[y:y+block_size, x:x+block_size]

        block_means.append(np.mean(block))

        hist = cv2.calcHist([block], [0], None, [256], [0, 256])
        histograms.append(hist)

    hist_avg = np.mean(histograms, axis=0)

    mu = float(np.mean(block_means))
    sigma = float(np.std(block_means))

    return hist_avg, mu, sigma


# ============================================================
# Gaussian adaptativo
# ============================================================
def apply_adaptive_gaussian_filter(image, sigma_noise, k=0.2):
    sigma_filter = max(0.1, sigma_noise * k)
    filtered = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_filter)
    return filtered, sigma_filter


# ============================================================
# 🔵 PIPELINE COMPLETO PARA UNA SOLA IMAGEN
# ============================================================
def process_single_image(image_array, block_size=20, num_blocks=60, k=0.08):
    """
    Procesa una única imagen (ya cargada en memoria como array numpy).
    """

    if image_array is None:
        raise ValueError("No se pudo cargar la imagen (image_array es None).")

    # Convertir a uint8 si hace falta
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    # 1. Obtener estadísticas de ruido
    _, mu, sigma = get_noise_stats_from_random_blocks(
        image_array, block_size=block_size, num_blocks=num_blocks
    )

    # 2. Calcular sigma adaptado
    sigma_adapted = max(sigma * k, 0.1)

    # 3. Aplicar Gaussian Blur
    processed_img = cv2.GaussianBlur(image_array, (0, 0), sigmaX=sigma_adapted)

    # 4. Métricas
    psnr_val, ssim_val, snr_val = compute_metrics(image_array, processed_img)

    snr_original = compute_snr(image_array.astype(np.float64), block_size=20)

    return {
        "original": image_array,
        "processed": processed_img,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "snr_original": snr_original,
        "snr_processed": snr_val,
        "sigma_adapted": sigma_adapted,
    }
