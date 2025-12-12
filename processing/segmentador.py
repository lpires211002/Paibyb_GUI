import cv2
import numpy as np
import tensorflow as tf

# Tamaño de entrada del modelo (ajustar si tu modelo usa otro)
SEG_IMG_SIZE = 256


# ==============================
# Cargar modelo de segmentación
# ==============================
def load_segmentation_model(model_path):
    """
    Carga un modelo Keras desde un archivo .h5
    """
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo: {e}")


# ==============================
# Preprocesamiento
# ==============================
def preprocess_image(img):
    """
    Prepara una imagen (uint8 en escala de grises) para el modelo.
    """
    img_resized = cv2.resize(img, (SEG_IMG_SIZE, SEG_IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=(0, -1))  # (1, h, w, 1)


# ==============================
# Postproceso
# ==============================
def postprocess_mask(mask_pred, orig_shape):
    """
    Redimensiona la máscara binaria al tamaño original.
    """
    mask = (mask_pred > 0.5).astype(np.float32)
    mask_resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
    return mask_resized


# ==============================
# Métricas básicas
# ==============================
def dice_score(mask_true, mask_pred):
    mask_true = mask_true.flatten()
    mask_pred = mask_pred.flatten()
    inter = np.sum(mask_true * mask_pred)
    return (2 * inter) / (np.sum(mask_true) + np.sum(mask_pred) + 1e-8)


def iou_score(mask_true, mask_pred):
    mask_true = mask_true.flatten()
    mask_pred = mask_pred.flatten()
    inter = np.sum(mask_true * mask_pred)
    union = np.sum(mask_true) + np.sum(mask_pred) - inter + 1e-8
    return inter / union


# ==============================
# Pipeline completo
# ==============================
def segment_image(model, image_np):
    """
    Devuelve:
    - máscara segmentada (tamaño original)
    - máscara predicha cruda
    """
    img_pre = preprocess_image(image_np)
    pred = model.predict(img_pre, verbose=0)[0, ..., 0]
    mask = postprocess_mask(pred, image_np.shape)
    return mask, pred
