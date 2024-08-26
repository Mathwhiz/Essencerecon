import cv2
import numpy as np
import pytesseract
import pygetwindow as gw
from pynput.keyboard import Controller, Key
import time
import os

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
custom_config = r'--oem 3 --psm 6 outputbase digits'

# Ruta a la carpeta de capturas y la de imágenes de referencia
screenshots_path = r'D:\_User Data\Documentos\My Games\Path of Exile\Screenshots'

# Coordenadas de la región de interés (ROI) en formato (x1, y1, x2, y2)
roi_first_row = (41, 147, 89, 725)
roi_last_row = (564, 147, 611, 721)

# Coordenadas de esencias en la columna izquierda
left_essences = {
    'greed': (3, 2),
    'contempt': (3, 51),
    'hatred': (3, 98),
    'woe': (3, 145),
    'fear': (3, 193),
    'anger': (3, 241),
    'torment': (3, 287),
    'sorrow': (3, 338),
    'rage': (3, 385),
    'suffering': (2, 434),
    'wrath': (3, 479),
    'doubt': (3, 530)
}

# Coordenadas de esencias en la columna derecha
right_essences = {
    'floating': (3, 2),
    'zeal': (2, 51),
    'anguish': (1, 98),
    'spite': (1, 145),
    'scorn': (1, 193),
    'envy': (2, 243),
    'missery': (2, 285),
    'dread': (3, 340),
    'insanity': (1, 385),
    'horror': (1, 434),
    'delirium': (1, 480),
    'hysteria': (1, 530)
}

# Esencias con parámetros específicos de preprocesamiento
custom_processing = {
    'contempt': {'contrast': True},
    'fear': {'threshold': True},
    'suffering': {'sharpen': True},
    'floating': {'contrast': True},
    'horror': {'contrast': True},
    'hysteria': {'threshold': True},
    'greed': {'contrast': True},
    'rage': {'contrast': True},
    'hatred': {'contrast': True},
    'spite': {'contrast': True},
    'missery': {'contrast': True}
}

def preprocess_image(image, essence_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rangos de color de referencia para cada esencia
    color_ranges = {
        
    }

    if essence_name in color_ranges:
        lower_bound = np.array(color_ranges[essence_name][0])
        upper_bound = np.array(color_ranges[essence_name][1])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(image, image, mask=mask)
        
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Aplicar el mismo preprocesamiento que para 'woe'
        result_gray = cv2.GaussianBlur(result_gray, (5, 5), 0)
        result_gray = cv2.adaptiveThreshold(result_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        return result_gray
    else:
        return image



def debug_preprocessing(image, essence_name):
    preprocessed_image = preprocess_image(image, essence_name)
    cv2.imshow(f"Preprocessed - {essence_name}", preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clean_text(text):
    return ''.join(filter(str.isdigit, text))

def extract_number(image, x, y, w, h, essence_name):
    number_region = image[y:y+h, x:x+w]
    
    # Probar diferentes factores de escalado
    number_region = cv2.resize(number_region, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    debug_preprocessing(number_region, essence_name)  # Llama aquí para depuración

    preprocessed_number_region = preprocess_image(number_region, essence_name)
    
    number_text = pytesseract.image_to_string(preprocessed_number_region, config=custom_config)
    cleaned_number_text = clean_text(number_text)
    print(f"Texto extraído: '{cleaned_number_text}' para esencia: {essence_name}")  # Para depuración
    
    try:
        number = int(cleaned_number_text) if cleaned_number_text else 0
        if number < 10 and cleaned_number_text:
            print(f"Advertencia: número muy bajo detectado '{cleaned_number_text}' para {essence_name}")
        if number > 2000:
            number = number // 10
        return number
    except ValueError:
        return 0


def process_image_for_essence(image, essence_name):
    preprocessed_image = preprocess_image(image, essence_name)
    text = extract_number(preprocessed_image)
    return text

def process_roi(cropped_img, essence_coords):
    essence_counts = {}
    for essence_name, (x, y) in essence_coords.items():
        width, height = 34, 22  # Ajusta si es necesario
        essence_count = extract_number(cropped_img, x, y, width, height, essence_name)
        essence_counts[essence_name] = essence_count
        cv2.rectangle(cropped_img, (x, y), (x + width, y + height), (0, 0, 255), 2)  # Rectángulo rojo
    return essence_counts, cropped_img

def find_latest_screenshot(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    if not files:
        return None

    files.sort(key=os.path.getmtime)
    return files[-1]

def activate_game_window():
    windows = gw.getWindowsWithTitle('Path of Exile')
    if windows:
        windows[0].activate()
        time.sleep(1)  # Esperar para asegurarse de que la ventana esté activa

def press_f8():
    keyboard = Controller()
    keyboard.press(Key.f8)
    time.sleep(0.1)  # Pequeña pausa para asegurarse de que la tecla se presione correctamente
    keyboard.release(Key.f8)

def main():
    activate_game_window()
    time.sleep(1)  # Espera para asegurarte de que la ventana esté activa

    # Presionar F8 para capturar la pantalla
    press_f8()
    
    time.sleep(2)  # Espera para asegurarte de que la captura se haya guardado

    latest_screenshot = find_latest_screenshot(screenshots_path)
    if not latest_screenshot:
        print("No se encontraron capturas de pantalla.")
        return

    img = cv2.imread(latest_screenshot)
    if img is None:
        print(f"No se pudo cargar la imagen: {latest_screenshot}")
        return

    # Procesar la primera fila
    first_row_crop = img[roi_first_row[1]:roi_first_row[3], roi_first_row[0]:roi_first_row[2]]
    left_counts, marked_img_first_row = process_roi(first_row_crop, left_essences)

    # Procesar la última fila
    last_row_crop = img[roi_last_row[1]:roi_last_row[3], roi_last_row[0]:roi_last_row[2]]
    right_counts, marked_img_last_row = process_roi(last_row_crop, right_essences)

    print("Conteo de esencias en la columna izquierda:", left_counts)
    print("Conteo de esencias en la columna derecha:", right_counts)

    # Mostrar las imágenes procesadas
    cv2.imshow("Primera fila procesada", marked_img_first_row)
    cv2.imshow("Última fila procesada", marked_img_last_row)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
