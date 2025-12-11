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