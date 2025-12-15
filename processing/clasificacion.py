import cv2
import numpy as np
import tensorflow as tf

# ==============================
# Configuración del clasificador
# ==============================
CLS_IMG_SIZE = 150  # ⚠️ debe coincidir con IMG_SIZE usado al entrenar
N_CLASSES = 4


# ==============================
# Cargar modelo de clasificación
# ==============================
def load_classification_model(model_path):
    """
    Carga el modelo de clasificación (.h5)
    """
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo clasificador: {e}")


# ==============================
# Preprocesamiento
# ==============================
def preprocess_for_classification(img):
    """
    img: numpy array uint8 (H, W) en escala de grises
    """
    img_resized = cv2.resize(img, (CLS_IMG_SIZE, CLS_IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = np.expand_dims(img_norm, axis=-1)   # (H, W, 1)
    img_norm = np.expand_dims(img_norm, axis=0)    # (1, H, W, 1)
    return img_norm


# ==============================
# Inferencia
# ==============================
def classify_image(model, image_np):
    img_pre = preprocess_for_classification(image_np)
    preds = model.predict(img_pre, verbose=0)[0]  # softmax
    class_idx = int(np.argmax(preds))
    return class_idx, preds
