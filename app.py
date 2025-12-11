import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# Importar tus dos métodos
from processing.GaussDenoise import process_single_image
from processing.NlmBilDenoise import process_nlm_pipeline


# ==============================
# Interfaz Streamlit
# ==============================
st.set_page_config(page_title="Interfaz de Filtrado de Imágenes", layout="wide")
st.title("Interfaz de Filtrado de Imágenes")


# ------------------------------
# Cargar archivo de imagen
# ------------------------------
uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["png", "jpg", "jpeg"]
)

# Selector del método
method = st.selectbox(
    "Método de filtrado",
    [
        "Gaussian Adaptive",
        "Bilateral + CLAHE + NLM"
    ]
)

# Parámetro de Gauss
k = None
if method == "Gaussian Adaptive":
    k = st.slider(
        "Coeficiente k (para Gaussian)",
        0.01, 0.20, 0.08, step=0.01
    )

# ------------------------------
# Procesamiento
# ------------------------------
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("L")  # Convertir a escala de grises
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

        else:  # NLM
            result = process_nlm_pipeline(
                image_array=image_np
            )

        # --------------------------
        # Mostrar resultados
        # --------------------------
        st.subheader("Imagen Procesada")
        st.image(result["processed"], clamp=True)

        st.subheader("Métricas")
        st.write(f"**PSNR:** {result['psnr']:.3f}")
        st.write(f"**SSIM:** {result['ssim']:.4f}")
        st.write(f"**SNR Original:** {result['snr_original']:.2f}")
        st.write(f"**SNR Procesado:** {result['snr_processed']:.2f}")

        # Parámetro de Gauss
        if method == "Gaussian Adaptive":
            st.write(f"**Sigma adaptado:** {result['sigma_adapted']:.4f}")


        # =====================================================
        # ✅ BOTÓN PARA DESCARGAR LA IMAGEN PROCESADA
        # =====================================================
        processed_img = result["processed"]
        buf = io.BytesIO()
        Image.fromarray(processed_img.astype(np.uint8)).save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="⬇️ Descargar imagen procesada",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png"
        )
        # =====================================================

    except Exception as e:
        st.error(f"Error procesando la imagen: {str(e)}")

else:
    st.info("Sube una imagen para comenzar.")
