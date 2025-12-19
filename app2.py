import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf

from processing import (
    process_single_image,
    process_nlm_pipeline,
    load_segmentation_model,
    segment_image,
    dice_score,
    iou_score,
    load_classification_model,
    classify_image,
    # Nuevas importaciones para extracción
    apply_harris,
    apply_sift,
    apply_orb,
    extraer_todas_caracteristicas,
    dibujar_contorno
)

# ==============================
# Configuración
# ==============================
SEG_MODEL_PATH = "modelos/segmentation.h5"
CLS_MODEL_PATH = "modelos/algoritmo_clasificador.h5"
CLASS_NAMES = ["Glioma", "Meningioma", "Sin Tumor", "Pituitaria"]

st.set_page_config(page_title="UI Procesamiento avanzado de imágenes médicas", layout="wide")

# ==============================
# CSS Personalizado - Estilo Reseda
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #fafafa;
    }
    
    h1 {
        font-size: 4.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #1a1a1a !important;
        margin-bottom: 1rem !important;
        text-transform: uppercase;
    }
    
    h2, h3 {
        font-weight: 600 !important;
        color: #2a2a2a !important;
        letter-spacing: -0.01em !important;
        margin-top: 2rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #e0e0e0;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #666666;
        font-weight: 500;
        font-size: 1rem;
        padding: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1a1a1a;
        border-bottom: 2px solid #1a1a1a;
    }
    
    [data-testid="stFileUploader"] {
        background-color: white;
        border: 2px dashed #d0d0d0;
        border-radius: 8px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #1a1a1a;
        background-color: #f8f8f8;
    }
    
    .stButton > button {
        background-color: #1a1a1a;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
        border-radius: 4px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        background-color: #2a2a2a;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stDownloadButton > button {
        background-color: #1a1a1a;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
        border-radius: 4px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        background-color: #2a2a2a;
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div {
        background-color: #666666;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
    }
    
    .stSlider {
        padding: 1rem 0;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    
    .stProgress > div > div {
        background-color: #1a1a1a;
    }
    
    [data-testid="stImage"] {
        border-radius: 4px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stAlert {
        background-color: white;
        border-left: 4px solid #1a1a1a;
        border-radius: 4px;
        padding: 1rem;
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    div[data-testid="column"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    p {
        color: #4a4a4a;
        line-height: 1.6;
    }
    
    .stSuccess {
        background-color: #f0f8f0;
        color: #2d5f2d;
        border-left: 4px solid #4caf50;
    }
    
    .stError {
        background-color: #fff5f5;
        color: #c62828;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

st.title("Procesamiento Avanzado de Imágenes Médicas UI")
st.subheader("Luca Pires - Magdalena La Valle")
st.markdown("##### Procesamiento y análisis de imágenes médicas")

# ==============================
# Cargar modelos
# ==============================
@st.cache_resource
def load_model_cached():
    try:
        return load_segmentation_model(SEG_MODEL_PATH)
    except Exception as e:
        st.error(f"Error cargando el modelo de segmentación: {e}")
        return None

model_seg = load_model_cached()

@st.cache_resource
def load_classifier_cached():
    try:
        return load_classification_model(CLS_MODEL_PATH)
    except Exception as e:
        st.error(f"Error cargando el modelo de clasificación: {e}")
        return None

model_cls = load_classifier_cached()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["PROCESAMIENTO", "SEGMENTACIÓN", "CLASIFICACIÓN", "EXTRACCIÓN DE CARACTERÍSTICAS"]
)

# =============================================================================
# TAB 1 — PROCESAMIENTO
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

            st.subheader("Imagen Procesada")
            st.image(result["processed"], clamp=True)

            st.subheader("Métricas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("PSNR", f"{result['psnr']:.3f}")
            with col2:
                st.metric("SSIM", f"{result['ssim']:.4f}")
            with col3:
                st.metric("SNR Original", f"{result['snr_original']:.2f}")
            with col4:
                st.metric("SNR Procesado", f"{result['snr_processed']:.2f}")

            if method == "Gaussian Adaptive":
                st.write(f"**Sigma adaptado:** {result['sigma_adapted']:.4f}")

            processed_img = result["processed"]
            buf = io.BytesIO()
            Image.fromarray(processed_img.astype(np.uint8)).save(buf, format="PNG")

            st.download_button(
                label="Descargar imagen procesada",
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
# TAB 2 — SEGMENTACIÓN
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

        try:
            mask, pred_raw = segment_image(model_seg, image_np)
        except Exception as e:
            st.error(f"Error durante segmentación: {e}")
            st.stop()

        st.subheader("Máscara Segmentada")
        st.image(mask * 255, clamp=True)

        overlay = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        overlay[..., 0] = np.maximum(overlay[..., 0], (mask * 255).astype(np.uint8))

        st.subheader("Overlay")
        st.image(overlay, clamp=True)

        if uploaded_seg_mask is not None:
            pil_mask = Image.open(uploaded_seg_mask).convert("L")
            mask_true = np.array(pil_mask)
            mask_true = (mask_true > 127).astype(np.float32)

            dice_val = dice_score(mask_true, mask)
            iou_val = iou_score(mask_true, mask)

            st.subheader("Métricas con Máscara Verdadera")
            if dice_val is not None and iou_val is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dice Score", f"{dice_val:.3f}")
                with col2:
                    st.metric("IoU Score", f"{iou_val:.3f}")
                if dice_val == 0 or iou_val == 0:
                    st.warning("Las métricas son cero. Verifica que la máscara verdadera corresponda a la imagen subida.")
        else:
            st.info("Sube una máscara verdadera para calcular métricas de evaluación.")

        buf = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")

        st.download_button(
            label="Descargar máscara",
            data=buf.getvalue(),
            file_name="segmentation_mask.png",
            mime="image/png"
        )

# =============================================================================
# TAB 3 — CLASIFICACIÓN
# =============================================================================
with tab3:
    st.markdown("### Clasificación de Imágenes")
    
    uploaded_cls = st.file_uploader(
        "Selecciona una imagen para clasificar",
        type=["png", "jpg", "jpeg"],
        key="cls"
    )

    if model_cls is None:
        st.error("❌ El modelo de clasificación no está cargado.")
    else:
        st.success("✅ Modelo de clasificación cargado correctamente.")

    if uploaded_cls is not None and model_cls is not None:
        try:
            pil_img = Image.open(uploaded_cls).convert("L")
            image_np = np.array(pil_img, dtype=np.uint8)
            
            st.subheader("Imagen Original")
            st.image(image_np, clamp=True, width=300)

            class_idx, probs = classify_image(model_cls, image_np)

            st.divider()
            
            st.subheader("Resultado de Clasificación")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    label="Clase Predicha",
                    value=CLASS_NAMES[class_idx],
                    delta=f"{probs[class_idx]*100:.1f}% confianza"
                )
            
            with col2:
                st.write("**Distribución de Probabilidades:**")
                for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(float(p))
                    with col_b:
                        st.write(f"{name}: **{p*100:.2f}%**")

        except Exception as e:
            import traceback
            st.error("ERROR EN CLASIFICACIÓN")
            st.code(traceback.format_exc())

# =============================================================================
# TAB 4 — EXTRACCIÓN DE CARACTERÍSTICAS
# =============================================================================
with tab4:
    st.markdown("### Extracción de Características")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        uploaded_extract_img = st.file_uploader(
            "Imagen original",
            type=["png", "jpg", "jpeg"],
            key="extract_img"
        )
    
    with col_upload2:
        uploaded_extract_mask = st.file_uploader(
            "Máscara segmentada",
            type=["png", "jpg", "jpeg"],
            key="extract_mask"
        )
    
    # Selector de método de extracción
    metodo_extraccion = st.selectbox(
        "Método de detección",
        ["Harris (Esquinas)", "SIFT (Características)", "ORB (Características)", "Métricas Geométricas"],
        key="metodo_extract"
    )
    
    if uploaded_extract_img is not None:
        # Cargar imagen original
        pil_img = Image.open(uploaded_extract_img).convert("L")
        img_original = np.array(pil_img, dtype=np.uint8)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen Original")
            st.image(img_original, clamp=True)
        
        # Aplicar detector seleccionado
        if metodo_extraccion == "Harris (Esquinas)":
            try:
                img_features = apply_harris(img_original)
                with col2:
                    st.subheader("Detección Harris")
                    st.image(img_features, clamp=True, channels="BGR")
                    st.caption("Esquinas detectadas en rojo")
            except Exception as e:
                st.error(f"Error aplicando Harris: {e}")
        
        elif metodo_extraccion == "SIFT (Características)":
            try:
                img_features = apply_sift(img_original)
                with col2:
                    st.subheader("Detección SIFT")
                    st.image(img_features, clamp=True, channels="BGR")
                    st.caption("Keypoints SIFT en verde")
            except Exception as e:
                st.error(f"Error aplicando SIFT: {e}")
        
        elif metodo_extraccion == "ORB (Características)":
            try:
                img_features = apply_orb(img_original)
                with col2:
                    st.subheader("Detección ORB")
                    st.image(img_features, clamp=True, channels="BGR")
                    st.caption("Keypoints ORB en azul")
            except Exception as e:
                st.error(f"Error aplicando ORB: {e}")
        
        elif metodo_extraccion == "Métricas Geométricas":
            if uploaded_extract_mask is not None:
                try:
                    # Cargar máscara
                    pil_mask = Image.open(uploaded_extract_mask).convert("L")
                    img_mask = np.array(pil_mask, dtype=np.uint8)
                    
                    # Binarizar máscara si es necesario
                    if img_mask.max() > 1:
                        img_mask = (img_mask > 127).astype(np.uint8) * 255
                    
                    with col2:
                        st.subheader("Máscara Segmentada")
                        st.image(img_mask, clamp=True)
                    
                    st.divider()
                    
                    # Extraer todas las características
                    features = extraer_todas_caracteristicas(img_mask, img_original)
                    
                    # Mostrar contorno
                    if features['contorno'] is not None:
                        img_contorno = dibujar_contorno(img_mask, features['contorno'])
                        st.subheader("Contorno Principal")
                        st.image(img_contorno, clamp=True, channels="BGR")
                    
                    # Mostrar métricas
                    st.subheader("Métricas Geométricas")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Circularidad", f"{features['circularidad']:.3f}" if not np.isnan(features['circularidad']) else "N/A")
                        st.metric("Extensión", f"{features['extension']:.3f}" if not np.isnan(features['extension']) else "N/A")
                    
                    with col_m2:
                        st.metric("Solidez", f"{features['solidez']:.3f}" if not np.isnan(features['solidez']) else "N/A")
                        st.metric("Elongación", f"{features['elongacion']:.3f}" if not np.isnan(features['elongacion']) else "N/A")
                    
                    with col_m3:
                        st.metric("Compacidad", f"{features['compacidad']:.3f}" if not np.isnan(features['compacidad']) else "N/A")
                        st.metric("Relación Áreas", f"{features['relacion_areas']:.3f}")
                    
                    st.divider()
                    
                    st.subheader("Información de Áreas")
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.metric("Área del Tumor", f"{features['area_tumor']} px²")
                    with col_a2:
                        st.metric("Área del Cráneo", f"{features['area_craneo']} px²")
                    
                except Exception as e:
                    import traceback
                    st.error("Error extrayendo características:")
                    st.code(traceback.format_exc())
            else:
                st.info("Para calcular métricas geométricas, sube también la máscara segmentada.")
    
    else:
        st.info("Sube una imagen para comenzar la extracción de características.")