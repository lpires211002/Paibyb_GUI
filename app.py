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
from processing.segmentador import load_segmentation_model, preprocess_image, postprocess_mask, dice_score, iou_score, segment_image


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

    uploaded_seg_mask = st.file_uploader(
        "Selecciona la mascara correspondiente para comparar la segmentación",
        type=["png", "jpg", "jpeg"],
        key="msk"
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
        

        if uploaded_seg_mask is not None:
            pil_mask = Image.open(uploaded_seg_mask).convert("L")
            mask_true = np.array(pil_mask)
            mask_true = (mask_true > 127).astype(np.float32)

            dice_val = dice_score(mask_true, mask)
            iou_val = iou_score(mask_true, mask)

            st.subheader("Métricas con Máscara Verdadera")
            if dice_val is not None and iou_val is not None:
                st.write(f"**Dice:** {dice_val:.3f}")
                st.write(f"**IoU:** {iou_val:.3f}")
                if dice_val == 0 or iou_val == 0:
                    st.warning("Las métricas son cero. Verifica que la máscara verdadera corresponda a la imagen subida.")
        else:
            st.info("Sube una máscara verdadera para calcular métricas de evaluación.")

       

        # Descarga
        buf = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")

        st.download_button(
            label="⬇️ Descargar máscara",
            data=buf.getvalue(),
            file_name="segmentation_mask.png",
            mime="image/png"
        )
