import cv2
import numpy as np
import os
from PIL import Image
import io

def scan_document(image_path):
    # Leer imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    original = image.copy()
    
    # Convertir a escala de grises y aplicar blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Usar un blur más fuerte como en el script de referencia
    blur = cv2.GaussianBlur(gray, (233, 233), 0)
    
    # Usar umbralización de Otsu en lugar de Canny
    _, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Encontrar contornos directamente del binario
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Usar el contorno más grande directamente
    doc_contour = max(contours, key=cv2.contourArea)
    
    # Aproximar al polígono
    epsilon = 0.02 * cv2.arcLength(doc_contour, True)
    doc_contour = cv2.approxPolyDP(doc_contour, epsilon, True)
    
    # Verificar que tengamos exactamente 4 puntos
    if len(doc_contour) != 4:
        # Ajustar epsilon hasta obtener 4 puntos
        epsilon = 0.01
        while len(doc_contour) != 4 and epsilon < 0.5:
            doc_contour = cv2.approxPolyDP(doc_contour, epsilon * cv2.arcLength(doc_contour, True), True)
            epsilon += 0.01
        
        if len(doc_contour) != 4:
            raise ValueError(f"No se pudo encontrar un contorno de 4 puntos. Puntos encontrados: {len(doc_contour)}")

    # Ordenar puntos usando norma de numpy
    points = doc_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # Top-Left
    rect[2] = points[np.argmax(s)]  # Bottom-Right

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # Top-Right
    rect[3] = points[np.argmax(diff)]  # Bottom-Left

    # Calcular dimensiones usando norma de numpy
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    
    # Puntos destino
    dst_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Transformar perspectiva
    matrix = cv2.getPerspectiveTransform(rect, dst_points)
    scanned = cv2.warpPerspective(original, matrix, (max_width, max_height))
    
    # Mejorar imagen
    gray_scanned = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(
        gray_scanned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9
    )
    
    # Después de obtener threshold, enderezar el texto
    def correct_skew(image):
        # Detectar bordes
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detectar líneas usando la transformada de Hough
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            # Calcular el ángulo dominante
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi
                # Normalizar ángulos alrededor de 0°, 90°, 180° o 270°
                if angle < 45:
                    angles.append(angle)
                elif angle < 135:
                    angles.append(angle - 90)
                else:
                    angles.append(angle - 180)
            
            # Tomar la mediana de los ángulos para mayor robustez
            median_angle = np.median(angles)
            
            # Rotar la imagen
            if abs(median_angle) > 0.5:  # Solo corregir si el ángulo es significativo
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
                return rotated
        
        return image

    # Aplicar la corrección de inclinación
    threshold = correct_skew(threshold)
    
    # Convertir la imagen procesada a formato PIL
    pil_image = Image.fromarray(threshold)
    return pil_image

# Uso
def process_documents():
    # Crear carpeta output si no existe
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Procesar todos los archivos en la carpeta input
    input_folder = 'input'
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join('output', f'{os.path.splitext(filename)[0]}.pdf')
            
            try:
                # Obtener imagen procesada en formato PIL
                pil_image = scan_document(input_path)
                # Guardar como PDF
                pil_image.save(output_path, 'PDF', resolution=100.0)
                print(f"Procesado exitosamente: {filename}")
            except Exception as e:
                print(f"Error procesando {filename}: {str(e)}")
                raise e

if __name__ == "__main__":
    process_documents()