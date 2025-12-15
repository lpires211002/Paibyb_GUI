import cv2
import numpy as np
import tensorflow as tf

# ==============================
# Configuración del clasificador
# ==============================
IMG_SIZE = 150  # Debe coincidir con el tamaño usado en el entrenamiento
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
        print(f"Modelo cargado exitosamente desde: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo clasificador: {e}")


# ==============================
# Preprocesar imagen para clasificación
# ==============================
def preprocess_for_classification(image_np):
    """
    Preprocesa una imagen para clasificación.
    
    Args:
        image_np: numpy array con la imagen en escala de grises (H, W)
    
    Returns:
        img_preprocessed: numpy array con forma (1, 150, 150, 1)
    """
    # Asegurar que sea numpy array
    if not isinstance(image_np, np.ndarray):
        raise TypeError(f"Se esperaba numpy array, se recibió: {type(image_np)}")
    
    # Asegurar que sea 2D (grayscale)
    if image_np.ndim == 3:
        # Si tiene 3 dimensiones, convertir a grayscale
        if image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        elif image_np.shape[2] == 1:
            image_np = image_np[:, :, 0]
    
    # Resize a IMG_SIZE x IMG_SIZE
    img_resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    
    # Mantener el rango 0-255 (sin normalización, igual que en tu notebook)
    # Solo asegurar que sea float32
    img_float = img_resized.astype(np.float32)
    
    # Reshape a (1, IMG_SIZE, IMG_SIZE, 1) para el modelo
    img_preprocessed = img_float.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return img_preprocessed


# ==============================
# Clasificar imagen
# ==============================
def classify_image(model, image_np):
    """
    Clasifica una imagen usando el modelo cargado.
    
    Args:
        model: Modelo de Keras cargado
        image_np: Imagen como numpy array (H, W) en escala de grises
    
    Returns:
        class_idx: Índice de la clase predicha (0-3)
        probabilities: Array con las probabilidades de cada clase
    """
    # Preprocesar imagen
    img_preprocessed = preprocess_for_classification(image_np)
    
    # Predecir probabilidades
    y_pred_probabilities = model.predict(img_preprocessed, verbose=0)
    
    # Obtener probabilidades (primera y única muestra del batch)
    probabilities = y_pred_probabilities[0]
    
    # Obtener índice de la clase con mayor probabilidad
    class_idx = int(np.argmax(probabilities))
    
    return class_idx, probabilities