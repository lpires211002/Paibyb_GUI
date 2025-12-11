import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
import processing

from processing import (
    process_single_image,
    process_nlm_pipeline,
)

print("Contenido del paquete:", dir(processing))

def load_segmentation_model(model_path):
    """
    Carga un modelo Keras desde un archivo .h5
    """
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo: {e}")

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



# ==============================
# Configuración
# ==============================
SEG_MODEL_PATH = "modelos/segmentation.h5"
SEG_IMG_SIZE = 256

st.set_page_config(page_title="Interfaz de Filtrado y Segmentación", layout="wide")
st.title("Interfaz de Filtrado y Segmentación de Imágenes")


# ==============================
# Cargar modelo de segmentación
# ==============================
@st.cache_resource
def load_model_cached():
    try:
        return load_segmentation_model(SEG_MODEL_PATH)
    except Exception as e:
        st.error(f"Error cargando el modelo de segmentación: {e}")
        return None

model_seg = load_model_cached()


# Tabs
tab1, tab2 = st.tabs(["🟦 Procesamiento", "🟥 Segmentación"])



# =============================================================================
# 🟦 TAB 1 — PROCESAMIENTO
# =============================================================================
with tab1:

    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["png", "jpg", "jpeg"],
        key="proc"
    )

    method = st.selectbox(
        "Método de filtrado",
        [
            "Gaussian Adaptive",
            "Bilateral + CLAHE + NLM"
        ],
        key="method_proc"
    )

    k = None
    if method == "Gaussian Adaptive":
        k = st.slider(
            "Coeficiente k (para Gaussian)",
            0.01, 0.20, 0.08, step=0.01,
            key="k_proc"
        )

    if uploaded_file is not None:

        pil_img = Image.open(uploaded_file).convert("L")
        image_np = np.array(pil_img)

        st.subheader("Imagen Original")
        st.image(image_np, clamp=True)
        st.divider()

        try:
            if method == "Gaussian Adaptive":
                result = process_single_image(
                    image_array=image_np,
                    k=k
                )
            else:
                result = process_nlm_pipeline(image_array=image_np)

            # Mostrar resultados
            st.subheader("Imagen Procesada")
            st.image(result["processed"], clamp=True)

            st.subheader("Métricas")
            st.write(f"**PSNR:** {result['psnr']:.3f}")
            st.write(f"**SSIM:** {result['ssim']:.4f}")
            st.write(f"**SNR Original:** {result['snr_original']:.2f}")
            st.write(f"**SNR Procesado:** {result['snr_processed']:.2f}")

            if method == "Gaussian Adaptive":
                st.write(f"**Sigma adaptado:** {result['sigma_adapted']:.4f}")

            # Descarga
            processed_img = result["processed"]
            buf = io.BytesIO()
            Image.fromarray(processed_img.astype(np.uint8)).save(buf, format="PNG")

            st.download_button(
                label="⬇️ Descargar imagen procesada",
                data=buf.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )

        except Exception as e:
            import traceback
            st.error("ERROR REAL:")
            st.code(traceback.format_exc())

    else:
        st.info("Sube una imagen para comenzar.")



# =============================================================================
# 🟥 TAB 2 — SEGMENTACIÓN
# =============================================================================
with tab2:

    uploaded_seg = st.file_uploader(
        "Selecciona una imagen para segmentar",
        type=["png", "jpg", "jpeg"],
        key="seg"
    )

    if model_seg is None:
        st.error("❌ El modelo no está cargado. Verifica la ruta modelos/segmentation.h5")
    else:
        st.success("✅ Modelo de segmentación cargado correctamente.")

    if uploaded_seg is not None and model_seg is not None:

        pil_img = Image.open(uploaded_seg).convert("L")
        image_np = np.array(pil_img)

        st.subheader("Imagen Original")
        st.image(image_np, clamp=True)

        # Ejecutar segmentación
        try:
            mask, pred_raw = segment_image(model_seg, image_np)
        except Exception as e:
            st.error(f"Error durante segmentación: {e}")
            st.stop()

        st.subheader("Máscara Segmentada")
        st.image(mask * 255, clamp=True)

        # Overlay
        overlay = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        overlay[..., 0] = np.maximum(overlay[..., 0], (mask * 255).astype(np.uint8))

        st.subheader("Overlay")
        st.image(overlay, clamp=True)

        # Métricas auto-evaluadas
        dice_val = dice_score(mask, mask)
        iou_val = iou_score(mask, mask)

        st.subheader("Métricas")
        st.write(f"**Dice:** {dice_val:.3f}")
        st.write(f"**IoU:** {iou_val:.3f}")

        # Descarga
        buf = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")

        st.download_button(
            label="⬇️ Descargar máscara",
            data=buf.getvalue(),
            file_name="segmentation_mask.png",
            mime="image/png"
        )
