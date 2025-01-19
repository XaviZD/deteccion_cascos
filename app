# Fase 1: Inicialización de recursos y configuración

from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify, send_from_directory
import cv2 
import pygame
from ultralytics import YOLO
import os
import atexit
import shutil
import numpy as np
import json
from sklearn.model_selection import KFold
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Inicializar la aplicación Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Rutas a los modelos
model_path_1 = r'C:/Users/PC/Desktop/sistema cascos v2/EPP-MODELO-cascos-nop-cascos.1-55EPP/runs/detect/train/weights/best.pt'  # Modelo general
model_path_2 = r'C:/Users/PC/Desktop/sistema cascos v2/EPP-MODELO-nuevo-sin-cascos.1-55EP/runs/detect/train/weights/best.pt'  # Modelo especializado

# Instanciar ambos modelos utilizando las rutas definidas
model1 = YOLO(model_path_1)
model2 = YOLO(model_path_2)

# Inicializar alarma
pygame.mixer.init()
pygame.mixer.music.load(r'C:/Users/PC/Desktop/sistema cascos v2/static/alarm.mp3')

# Crear carpetas para guardar recortes de detecciones si no existen
recortes_dir = "C:/Users/PC/Desktop/sistema cascos v2/recortes"
recortes_con_casco = os.path.join(recortes_dir, "con_casco")
recortes_sin_casco = os.path.join(recortes_dir, "sin_casco")
os.makedirs(recortes_con_casco, exist_ok=True)
os.makedirs(recortes_sin_casco, exist_ok=True)

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

# Variable de suavizado temporal para detecciones consecutivas
deteccion_previa = {"con_casco": 0, "sin_casco": 0}

# Función para limpiar el directorio de recortes al finalizar
def limpiar_recortes():
    shutil.rmtree(recortes_con_casco, ignore_errors=True)
    shutil.rmtree(recortes_sin_casco, ignore_errors=True)


atexit.register(limpiar_recortes)

# Fase 2: Detección en Tiempo Real

def generate_frames():
    frame_counter = 0  # Contador de frames para nombrar los archivos de imágenes
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_resized = cv2.resize(frame, (640, 480))
        results_model1 = model1(frame_resized)

        con_casco_detectado = False
        sin_casco_detectado = False

        for result in results_model1:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                class_id = int(box.cls)

                if class_id == 0 and confidence > 0.7:
                    crop_img = frame_resized[y1:y2, x1:x2]
                    results_model2 = model2(crop_img)
                    casco_confirmado = True

                    for specialized_result in results_model2:
                        for specialized_box in specialized_result.boxes:
                            if specialized_box.conf.item() >= 0.99:
                                casco_confirmado = False
                                break

                    if casco_confirmado:
                        label = 'Con casco'
                        color = (0, 255, 0)
                        con_casco_detectado = True
                        deteccion_previa["con_casco"] += 1
                        cv2.imwrite(os.path.join(recortes_con_casco, f"frame_{frame_counter}.jpg"), crop_img)
                    else:
                        label = 'Sin casco (corregido)'
                        color = (0, 0, 255)
                        sin_casco_detectado = True
                        deteccion_previa["sin_casco"] += 1
                        cv2.imwrite(os.path.join(recortes_sin_casco, f"frame_{frame_counter}.jpg"), crop_img)

                    text_y = y1 - 10 if y1 > 30 else y2 + 20
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, f"{label} ({confidence:.2f})", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                elif class_id == 1 and confidence > 0.6:
                    label = 'Sin casco'
                    color = (0, 0, 255)
                    sin_casco_detectado = True
                    deteccion_previa["sin_casco"] += 1
                    crop_img = frame_resized[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(recortes_sin_casco, f"frame_{frame_counter}.jpg"), crop_img)
                    text_y = y1 - 10 if y1 > 30 else y2 + 20
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, f"{label} ({confidence:.2f})", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if sin_casco_detectado and deteccion_previa["sin_casco"] > 3:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()

        deteccion_previa["sin_casco"] = 0 if not sin_casco_detectado else deteccion_previa["sin_casco"]
        deteccion_previa["con_casco"] = 0 if not con_casco_detectado else deteccion_previa["con_casco"]

        ret, buffer = cv2.imencode('.jpg', frame_resized)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        frame_counter += 1

# Fase 3: Gestión de Rutas y Plantillas

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camara')
def camara():
    return render_template('camara.html')

@app.route('/modelo')
def modelo():
    return render_template('modelo.html')

@app.route('/configurar')
def configurar():
    return render_template('configuracion.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Fase 4: Guardado de entrenamientos y listado de los “best.pt”

# Función para crear la carpeta principal de entrenamientos
def create_training_folder(base_path):
    # Obtener números de carpetas existentes y determinar la siguiente
    existing_folders = [int(folder.split('_')[-1]) for folder in os.listdir(base_path) if folder.startswith("entrenamiento")]
    next_folder_number = max(existing_folders, default=0) + 1
    training_folder_path = os.path.join(base_path, f"entrenamiento_{next_folder_number}")
    os.makedirs(training_folder_path, exist_ok=True)
    return training_folder_path

# Rutas para guardar resultados anteriores
last_training_file = "C:/Users/PC/Desktop/sistema cascos v2/last_training_results.json"

# Función para guardar resultados del último entrenamiento
def save_last_training_results(results):
    with open(last_training_file, 'w') as f:
        json.dump(results, f)

# Función para cargar los resultados del último entrenamiento
def load_last_training_results():
    if os.path.exists(last_training_file):
        with open(last_training_file, 'r') as f:
            return json.load(f)
    return None


def guardar_modelo_best(modelo_path, destino_path):
    """
    Guarda un archivo de modelo 'best.pt' en una carpeta destino con un nombre secuencial.
    
    Args:
        modelo_path (str): Ruta al archivo del modelo a guardar.
        destino_path (str): Ruta de la carpeta donde se guardarán los modelos.
    """
    # Crear la carpeta destino si no existe
    os.makedirs(destino_path, exist_ok=True)

    # Obtener todos los archivos existentes en la carpeta de destino
    archivos_existentes = [
        f for f in os.listdir(destino_path)
        if f.startswith("entrenamiento-") and f.endswith(".pt")
    ]

    # Extraer números de los nombres de los archivos
    numeros_existentes = [
        int(f.split('-')[1].split('.')[0]) for f in archivos_existentes if f.split('-')[1].split('.')[0].isdigit()
    ]

    # Determinar el siguiente número disponible
    siguiente_numero = max(numeros_existentes) + 1 if numeros_existentes else 1

    # Crear el nuevo nombre del archivo
    nuevo_nombre = f"entrenamiento-{siguiente_numero}.pt"
    destino_completo = os.path.join(destino_path, nuevo_nombre)

    # Mover el archivo al destino
    shutil.move(modelo_path, destino_completo)  # Usa shutil.copy si prefieres copiar

    print(f"Archivo guardado como: {destino_completo}")

@app.route('/get_best_entrenamientos', methods=['GET'])
def get_best_entrenamientos():
    best_entrenamientos_dir = r"C:/Users/PC/Desktop/sistema cascos v2/best_entrenamientos"
    # Filtramos solo los archivos .pt
    best_entrenamientos = [f for f in os.listdir(best_entrenamientos_dir) if f.endswith('.pt')]
    
    # Si no hay archivos .pt, retornar un mensaje de error
    if not best_entrenamientos:
        return jsonify({'error': 'No se encontraron archivos .pt'}), 404
    
    return jsonify(best_entrenamientos)


# Fase 5: Funcion de entrenamiento

# Ruta para el entrenamiento
@app.route('/entrenar', methods=['GET', 'POST'])
def entrenar():
    last_results = load_last_training_results()

    # Obtener la lista de los entrenamientos previos
    best_models_folder = r"C:/Users/PC/Desktop/sistema cascos v2/best_entrenamientos"
    entrenamientos_previos = []
    if os.path.exists(best_models_folder):
        entrenamientos_previos = [d for d in os.listdir(best_models_folder) if os.path.isdir(os.path.join(best_models_folder, d))]

    if request.method == 'POST':
        epochs = int(request.form.get("epochs", 150))

        # Obtener el correo del destinatario desde el formulario
        recipient_email = request.form.get("recipient_email")

        # Definir rutas
        yaml_file = r"C:/Users/PC/Desktop/sistema cascos v2/prueba_entrenamiento\data.yaml"
        model_path = r"C:/Users/PC/Desktop/sistema cascos v2/EPP-MODELO-cascos-nop-cascos.1-55EPP/yolov8l.pt"
        
        # Definir carpetas de imágenes y etiquetas
        train_images_dir = r"C:/Users/PC/Desktop/sistema cascos v2/prueba_entrenamiento/train/images"
        valid_images_dir = r"C:/Users/PC/Desktop/sistema cascos v2/prueba_entrenamiento/valid/images"
        train_labels_dir = r"C:/Users/PC/Desktop/sistema cascos v2/prueba_entrenamiento/train/labels"
        valid_labels_dir = r"C:/Users/PC/Desktop/sistema cascos v2/prueba_entrenamiento/valid/labels"

        # Crear listas de imágenes y etiquetas
        train_images = [os.path.join(train_images_dir, f) for f in os.listdir(train_images_dir)]
        valid_images = [os.path.join(valid_images_dir, f) for f in os.listdir(valid_images_dir)]
        train_labels = [os.path.join(train_labels_dir, f) for f in os.listdir(train_labels_dir)]
        valid_labels = [os.path.join(valid_labels_dir, f) for f in os.listdir(valid_labels_dir)]
        
        # Combinar datos de train y valid
        images = np.array(train_images + valid_images)
        labels = np.array(train_labels + valid_labels)

        precision_list, recall_list, map50_list, map_list = [], [], [], []

        # Crear la carpeta de entrenamiento principal
        base_training_folder = "C:/Users/PC/Desktop/sistema cascos v2/entrenamientos"
        main_training_folder = create_training_folder(base_training_folder)

        # Dividir datos con KFold y realizar entrenamiento en cada ciclo
        kf = KFold(n_splits=5, shuffle=True)
        for fold_index, (train_index, valid_index) in enumerate(kf.split(images), 1):
            cycle_folder = os.path.join(main_training_folder, f"ciclo_{fold_index}")
            os.makedirs(cycle_folder, exist_ok=True)

            model = YOLO(model_path)
            model.train(data=yaml_file, epochs=epochs, imgsz=640, batch=8, workers=0, project=cycle_folder, name="train")

            results = model.val(data=yaml_file)
            precision_list.append(results.box.mp)
            recall_list.append(results.box.mr)
            map50_list.append(results.box.map50)
            map_list.append(results.box.map)

            if fold_index == 5:
                # Guardar modelo final utilizando guardar_modelo_best
                best_model_source = os.path.join(cycle_folder, "train", "weights", "best.pt")
                guardar_modelo_best(best_model_source, best_models_folder)
                enviar_correo_notificacion(
                    recipient_email,  # Enviar el correo a la dirección proporcionada
                    avg_precision = round(np.mean(precision_list), 2), 
                    avg_recall = round(np.mean(recall_list), 2), 
                    avg_map50 = round(np.mean(map50_list), 2), 
                    avg_map = round(np.mean(map_list), 2)
                )


        avg_precision = round(np.mean(precision_list), 2)
        avg_recall = round(np.mean(recall_list), 2)
        avg_map50 = round(np.mean(map50_list), 2)
        avg_map = round(np.mean(map_list), 2)

        # Guardar resultados actuales como "anteriores" para la próxima vez
        current_results = {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_map50": avg_map50,
            "avg_map": avg_map
        }
        save_last_training_results(current_results)

        # Comparar con el entrenamiento anterior
        comparison = {}
        if last_results:
            comparison['precision'] = "Mejor" if avg_precision > last_results["avg_precision"] else "Peor" if avg_precision < last_results["avg_precision"] else "Igual"
            comparison['recall'] = "Mejor" if avg_recall > last_results["avg_recall"] else "Peor" if avg_recall < last_results["avg_recall"] else "Igual"
            comparison['map50'] = "Mejor" if avg_map50 > last_results["avg_map50"] else "Peor" if avg_map50 < last_results["avg_map50"] else "Igual"
            comparison['map'] = "Mejor" if avg_map > last_results["avg_map"] else "Peor" if avg_map < last_results["avg_map"] else "Igual"


        return render_template(
            'entrenamiento.html', 
            avg_precision=avg_precision, avg_recall=avg_recall, 
            avg_map50=avg_map50, avg_map=avg_map, 
            last_results=last_results, comparison=comparison, 
            entrenamientos_previos=entrenamientos_previos
        )
    # Ruta al directorio de entrenamientos
    entrenamientos_dir = r"C:/Users/PC/Desktop/sistema cascos v2/entrenamientos"
    entrenamientos_list = [d for d in os.listdir(entrenamientos_dir) if os.path.isdir(os.path.join(entrenamientos_dir, d))]
    

    return render_template('entrenamiento.html', last_results=last_results, entrenamientos_previos=entrenamientos_previos, entrenamientos=entrenamientos_list)

# Fase 6: Validacion  y envio de correo

# Función para validar el correo
def validate_email(email):
    # Verifica si el correo contiene '@gmail.com'
    return email.endswith('@gmail.com')or email.endswith('@utm.edu.ec')or email.endswith('@hotmail.com')

@app.route('/asignar_correo', methods=['POST'])
def asignar_correo():
    # Obtener los datos enviados desde el cliente en formato JSON
    data = request.get_json()
    email = data.get('email', '')  # Obtener el correo o dejar vacío por defecto

    if email:
        # Validar si el correo es un correo de Gmail
        if not validate_email(email):
            return jsonify({'status': 'error', 'message': 'No es posible asignar. El correo debe ser un correo electronico valido.'}), 400
        
        # Si el correo es válido, enviar mensaje de éxito
        message = f"Correo asignado: {email}"
        return jsonify({'status': 'success', 'message': message})

    else:
        # Si el correo está vacío, enviar mensaje de error
        message = "No se asignó correo."
        return jsonify({'status': 'error', 'message': message}), 400


# Modificar la función de envío de correo
def enviar_correo_notificacion(recipient_email, avg_precision, avg_recall, avg_map50, avg_map):

    sender_email = "sistema.deteccion.24@gmail.com" 
    sender_password = "qrym butz enrz yjwk" 

    subject = "Entrenamiento completado"
    body = f"""
    El entrenamiento ha finalizado exitosamente.

    Resultados:
    - Precisión: {avg_precision}
    - Recall: {avg_recall}
    - maAP@50: {avg_map50}
    - mAP: {avg_map}

    Atentamente,
    Tu sistema de detección
    """

    try:
        # Configurar el correo
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Enviar el correo
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("Correo enviado exitosamente.")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")

# Fase 7: Cargar el entrenamiento en el sistema de deteccion 

# Ruta para cargar el entrenamiento seleccionado y copiarlo
@app.route('/cargar_entrenamiento', methods=['POST'])
def cargar_entrenamiento():
    # Ruta de origen y destino
    base_dir = 'C:/Users/PC/Desktop/sistema cascos v2/best_entrenamientos'
    destino_dir = 'C:/Users/PC/Desktop/sistema cascos v2/EPP-MODELO-cascos-nop-cascos.1-55EPP/runs/detect/train/weights'

    # Obtener el archivo seleccionado del formulario
    entrenamiento = request.form.get('entrenamiento')
    if not entrenamiento:
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    # Rutas completas
    origen_path = os.path.join(base_dir, entrenamiento)
    destino_path = os.path.join(destino_dir, 'best.pt')  # Se renombrará como best.pt

    try:
        # Copiar y renombrar el archivo
        shutil.copy2(origen_path, destino_path)
        return jsonify({'message': f'Archivo {entrenamiento} cargado exitosamente.'}), 200
    except FileNotFoundError:
        return jsonify({'error': 'Archivo no encontrado en la carpeta de origen.'}), 404
    except Exception as e:
        return jsonify({'error': f'Error al copiar el archivo: {e}'}), 500


# Ruta para descargar el entrenamiento seleccionado
@app.route('/descargar_entrenamiento', methods=['POST'])
def descargar_entrenamiento():
    # Carpeta donde están almacenados los entrenamientos
    base_dir = 'C:/Users/PC/Desktop/sistema cascos v2/best_entrenamientos'

    # Obtener el archivo seleccionado desde el parámetro de consulta
    entrenamiento = request.form.get('entrenamiento')
    if not entrenamiento:
        return jsonify({'error': 'No se especificó ningún archivo para descargar'}), 400

    # Ruta completa del archivo
    origen_path = os.path.join(base_dir, entrenamiento)

    try:
        # Validar que el archivo exista
        if not os.path.exists(origen_path):
            return jsonify({'error': 'Archivo no encontrado'}), 404

        # Enviar el archivo al usuario para su descarga
        return send_from_directory(directory=base_dir, path=entrenamiento, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Error al intentar descargar el archivo: {e}'}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))