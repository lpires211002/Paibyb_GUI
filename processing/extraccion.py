"""
Módulo de extracción de características de imágenes médicas
Incluye detectores de características (Harris, SIFT, ORB) y métricas geométricas
"""

import cv2
import numpy as np


# =============================================================================
# DETECTORES DE CARACTERÍSTICAS
# =============================================================================

def apply_harris(img, block_size=2, ksize=3, k=0.04, thresh=0.01):
    """
    Aplica el detector de esquinas Harris a una imagen.
    
    Args:
        img: Imagen en escala de grises (numpy array)
        block_size: Tamaño del vecindario considerado
        ksize: Parámetro de apertura del kernel Sobel
        k: Parámetro libre de Harris
        thresh: Umbral relativo para detectar esquinas
        
    Returns:
        img_color: Imagen BGR con esquinas marcadas en rojo
    """
    # Asegurar tipo float32
    gray = np.float32(img)
    
    # Calcular respuesta de Harris
    harris_response = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # Dilatar para hacer más visibles las esquinas
    harris_response = cv2.dilate(harris_response, None)
    
    # Crear imagen color (BGR) para marcar esquinas
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Umbral: esquinas donde la respuesta supera cierto porcentaje
    corners = harris_response > thresh * harris_response.max()
    
    # Marcar esquinas en rojo (BGR)
    img_color[corners] = [0, 0, 255]
    
    return img_color


def apply_sift(img, nfeatures=20, contrast_threshold=0.05, edge_threshold=10, sigma=2):
    """
    Aplica el detector SIFT (Scale-Invariant Feature Transform).
    
    Args:
        img: Imagen en escala de grises
        nfeatures: Número de características a detectar
        contrast_threshold: Umbral de contraste
        edge_threshold: Umbral de bordes
        sigma: Sigma del Gaussiano
        
    Returns:
        img_output: Imagen con keypoints SIFT dibujados en verde
    """
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )
    
    kp = sift.detect(img, None)
    
    img_output = cv2.drawKeypoints(
        img,
        kp,
        None,
        (0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    return img_output


def apply_orb(img, nfeatures=20, scale_factor=2, nlevels=8, edge_threshold=31):
    """
    Aplica el detector ORB (Oriented FAST and Rotated BRIEF).
    
    Args:
        img: Imagen en escala de grises
        nfeatures: Número de características a detectar
        scale_factor: Factor de escala de la pirámide
        nlevels: Número de niveles en la pirámide
        edge_threshold: Tamaño del borde
        
    Returns:
        img_output: Imagen con keypoints ORB dibujados en azul
    """
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=scale_factor,
        nlevels=nlevels,
        edgeThreshold=edge_threshold,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31
    )
    
    kp, descriptors = orb.detectAndCompute(img, None)
    
    img_output = cv2.drawKeypoints(
        img,
        kp,
        None,
        color=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    
    return img_output


# =============================================================================
# FUNCIONES AUXILIARES PARA MÉTRICAS
# =============================================================================

def contar_area(img):
    """
    Cuenta el número de píxeles blancos (255) en una imagen binaria.
    
    Args:
        img: Imagen binaria
        
    Returns:
        area: Número de píxeles blancos
    """
    return np.sum(img == 255)


def binarizar_imagen(img, umbral=35):
    """
    Binariza una imagen en escala de grises.
    
    Args:
        img: Imagen en escala de grises
        umbral: Valor de umbral para binarización
        
    Returns:
        img_bin: Imagen binarizada
    """
    _, img_bin = cv2.threshold(
        img,
        umbral,
        255,
        cv2.THRESH_BINARY
    )
    return img_bin


def calcular_relacion_areas(img_segmentada, img_original, umbral=35):
    """
    Calcula la relación entre el área del tumor y el área del cráneo.
    
    Args:
        img_segmentada: Máscara binaria del tumor
        img_original: Imagen original en escala de grises
        umbral: Umbral para binarizar la imagen original
        
    Returns:
        relacion: Relación área_tumor / área_cráneo
        area_tumor: Área del tumor en píxeles
        area_craneo: Área del cráneo en píxeles
    """
    # Área del tumor
    area_tumor = contar_area(img_segmentada)
    
    # Área de la cabeza
    original_binarizada = binarizar_imagen(img_original, umbral)
    area_craneo = contar_area(original_binarizada)
    
    # Calcular la relación (evitar división por cero)
    relacion = area_tumor / area_craneo if area_craneo > 0 else 0
    
    return relacion, area_tumor, area_craneo


# =============================================================================
# MÉTRICAS GEOMÉTRICAS DE CONTORNOS
# =============================================================================

def calcular_metricas_contorno(contorno):
    """
    Calcula métricas geométricas de un contorno.
    
    Args:
        contorno: Contorno de OpenCV
        
    Returns:
        circularidad: Qué tan circular es el contorno (1 = círculo perfecto)
        extension: Proporción del rectángulo delimitador ocupado
        solidez: Proporción del casco convexo ocupado
        elongacion: Relación entre eje mayor y menor
        compacidad: Relación perímetro²/área
    """
    # Validar que el contorno tenga suficientes puntos
    if contorno is None or len(contorno) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Calcular métricas básicas
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    x, y, w, h = cv2.boundingRect(contorno)
    
    # Casco convexo
    env_convexa = cv2.convexHull(contorno)
    area_env_convexa = cv2.contourArea(env_convexa)
    
    # Elipse ajustada
    (x_c, y_c), (eje_mayor, eje_menor), theta = cv2.fitEllipse(contorno)
    
    # Calcular métricas (con protección contra división por cero)
    circularidad = 4 * np.pi * area / (perimetro**2) if perimetro > 0 else np.nan
    extension = area / (w * h) if (w * h) > 0 else np.nan
    solidez = area / area_env_convexa if area_env_convexa > 0 else np.nan
    elongacion = eje_mayor / eje_menor if eje_menor > 0 else np.nan
    compacidad = perimetro**2 / area if area > 0 else np.nan
    
    return circularidad, extension, solidez, elongacion, compacidad


def extraer_contorno_principal(img_binaria):
    """
    Extrae el contorno más grande de una imagen binaria.
    
    Args:
        img_binaria: Imagen binaria (máscara)
        
    Returns:
        contorno: Contorno más grande encontrado (o None si no hay)
    """
    cnts, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) == 0:
        return None
    
    # Retornar el contorno con mayor área
    contorno_principal = max(cnts, key=cv2.contourArea)
    return contorno_principal


def dibujar_contorno(img_binaria, contorno, color=(255, 0, 0), grosor=2):
    """
    Dibuja un contorno sobre una imagen.
    
    Args:
        img_binaria: Imagen binaria base
        contorno: Contorno a dibujar
        color: Color BGR del contorno
        grosor: Grosor de la línea
        
    Returns:
        img_contorno: Imagen RGB con el contorno dibujado
    """
    if contorno is None:
        # Si no hay contorno, devolver imagen convertida a RGB
        return cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)
    
    img_rgb = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)
    img_contorno = cv2.drawContours(img_rgb, [contorno], -1, color, grosor)
    
    return img_contorno


# =============================================================================
# FUNCIÓN PRINCIPAL DE EXTRACCIÓN DE CARACTERÍSTICAS
# =============================================================================

def extraer_todas_caracteristicas(img_segmentada, img_original):
    """
    Extrae todas las características de una imagen segmentada.
    
    Args:
        img_segmentada: Máscara binaria del tumor (0 o 255)
        img_original: Imagen original en escala de grises
        
    Returns:
        features: Diccionario con todas las características extraídas
    """
    # Extraer contorno principal
    contorno = extraer_contorno_principal(img_segmentada)
    
    # Calcular métricas geométricas
    circ, ext, sol, elong, comp = calcular_metricas_contorno(contorno)
    
    # Calcular áreas
    relacion, area_tumor, area_craneo = calcular_relacion_areas(img_segmentada, img_original)
    
    features = {
        'circularidad': circ,
        'extension': ext,
        'solidez': sol,
        'elongacion': elong,
        'compacidad': comp,
        'area_tumor': area_tumor,
        'area_craneo': area_craneo,
        'relacion_areas': relacion,
        'contorno': contorno
    }
    
    return features