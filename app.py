import os
import torch
import tempfile
from flask import Flask, request, jsonify
import psycopg2
from ultralytics import YOLO
from datetime import datetime
import requests
import traceback
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURACIÓN SEGURA CON VARIABLES DE ENTORNO ---
IMG_BB_KEY = os.getenv("IMG_BB_KEY")
DB_URL = os.getenv("DB_URL")
NODE_URL = os.getenv("NODE_URL")
PORT = int(os.getenv("PORT"))

# Archivo de pesos entrenado para la detección de EPPs
ARCHIVO_MODELO = "Epp_RN3.pt"

# PARCHE DE COMPATIBILIDAD DE PYTORCH
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

app = Flask(__name__)

# CARGA DEL MODELO NEURONAL
if os.path.exists(ARCHIVO_MODELO):
    print("Cargando Red Neuronal EPP (Modelo S.E.P.P. RN3)...")
    model = YOLO(ARCHIVO_MODELO, task="detect")
    print("Modelo cargado correctamente.")
else:
    model = None
    print(f"ERROR: Modelo no encontrado: {ARCHIVO_MODELO}")

# TRADUCCIÓN DE CLASES DEL NUEVO MODELO
TRADUCCION_CLASES = {
    "glove": "guantes",
    "goggles": "lentes",
    "head": "cabeza",
    "helmet": "casco",
    "mask": "mascarilla",
    "no-goggles": "sin_lentes",
    "no_glove": "sin_guantes",
    "no_goggles": "sin_lentes",
    "no_mask": "sin_mascarilla",
    "person": "persona",
    "vest": "chaleco"
}

def obtener_conexion():
    return psycopg2.connect(DB_URL)

# FUNCIÓN DE INTEGRACIÓN CON IMGBB
# Sube la evidencia fotográfica a un repositorio en la nube.
def subir_a_imgbb(ruta_imagen):
    try:
        with open(ruta_imagen, "rb") as file:
            payload = {"key": IMG_BB_KEY}
            files = {"image": file}

            res = requests.post(
                "https://api.imgbb.com/1/upload",
                data=payload,
                files=files,
                timeout=20
            )

            if res.status_code == 200:
                return res.json()['data']['url']
            else:
                print(f"Error en ImgBB ({res.status_code}): {res.text}")
                return "https://via.placeholder.com/150?text=Error+Subida"

    except Exception as e:
        print(f"ERROR CRÍTICO subiendo a ImgBB: {e}")
        return "https://via.placeholder.com/150?text=Error+Critico"


# ENDPOINT: LISTADO DE CÁMARAS Y SALONES
@app.route('/obtener_camaras', methods=['GET'])
def obtener_camaras():
    conn = None
    cur = None

    try:
        conn = obtener_conexion()
        cur = conn.cursor()

        # Consulta relacional para obtener los códigos de cámara vinculados a su salón.
        cur.execute("""
            SELECT c.codigo_camara, s.nombre_salon
            FROM camaras c
            JOIN salones s ON c.id_salon = s.id_salon
        """)

        filas = cur.fetchall()

        return jsonify([
            f"{fila[0]} ({fila[1]})" for fila in filas
        ]), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# ENDPOINT PRINCIPAL: SIMULACIÓN Y ANÁLISIS DE IA
@app.route('/simular_camara', methods=['POST'])
def simular_camara():

    if model is None:
        return jsonify({
            "error": "El motor de IA no está cargado"
        }), 500

    codigo_camara_raw = request.form.get("codigo_camara")
    foto_archivo = request.files.get('imagen')

    if not codigo_camara_raw or not foto_archivo:
        return jsonify({"error": "Faltan datos"}), 400

    # Limpieza de la cadena para aislar el código de la cámara
    codigo_camara = codigo_camara_raw.split(" ")[0].strip()

    ruta_temporal = None
    conn = None
    cur = None
    id_docente = None

    try:
        # Almacenamiento temporal de la imagen recibida
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp_file:
            ruta_temporal = temp_file.name

        foto_archivo.save(ruta_temporal)

        conn = obtener_conexion()
        cur = conn.cursor()

        dias = {
            0: "Lunes",
            1: "Martes",
            2: "Miércoles",
            3: "Jueves",
            4: "Viernes",
            5: "Sábado",
            6: "Domingo"
        }

        dia_actual = dias[datetime.now().weekday()].strip().capitalize()
        hora_actual = datetime.now().time()

        # Cruce de horarios para identificar al docente responsable
        cur.execute("""
            SELECT h.id_docente
            FROM horarios_clases h
            JOIN camaras c ON h.id_salon = c.id_salon
            WHERE c.codigo_camara = %s
            AND h.dia_semana = %s
            AND %s BETWEEN h.hora_inicio AND h.hora_fin
        """, (codigo_camara, dia_actual, hora_actual))

        horario_reg = cur.fetchone()
        id_docente = horario_reg[0] if horario_reg else None

        # INFERENCIA CON YOLO
        results = model.predict(
            source=ruta_temporal,
            conf=0.25,
            save=False,
            verbose=False
        )

        print("\n========== RESULTADOS YOLO S.E.P.P. ==========")

        # Diccionario para contar las clases en español
        conteos = {
            'persona': 0,
            'cabeza': 0,
            'casco': 0,
            'guantes': 0,
            'lentes': 0,
            'mascarilla': 0,
            'chaleco': 0,
            'sin_guantes': 0,
            'sin_lentes': 0,
            'sin_mascarilla': 0
        }

        # Traducción y conteo de las detecciones
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            nombre_original = str(model.names[cls_id]).lower().strip()
            confianza = float(box.conf.item())

            # Traducción del nombre original al español
            nombre = TRADUCCION_CLASES.get(
                nombre_original,
                nombre_original
            )

            print(
                f"Clase original: {nombre_original} | "
                f"Clase: {nombre} | "
                f"Confianza: {confianza:.4f}"
            )

            if nombre in conteos:
                conteos[nombre] += 1

        print("==============================================")

        print("\n========== CONTEOS S.E.P.P. ==========")
        for clase, cantidad in conteos.items():
            print(f"{clase}: {cantidad}")
        print("======================================")

        personas = conteos['persona']

        # CASO A: Control si no hay personas en el campo visual
        if personas == 0:
            print("No se detectaron personas en el área.")

            return jsonify({
                "estado": "Procesado",
                "infraccion": False,
                "motivo": "No se detectaron personas",
                "docente_notificado": False
            }), 200

        # VALIDACIÓN DE REGLAS E INFRACCIONES
        faltas = []

        # A. Conteo directo de clases negativas disponibles en RN3

        if conteos['sin_guantes'] > 0:
            faltas.append(
                f"{conteos['sin_guantes']} sin guantes"
            )

        if conteos['sin_lentes'] > 0:
            faltas.append(
                f"{conteos['sin_lentes']} sin lentes"
            )

        if conteos['sin_mascarilla'] > 0:
            faltas.append(
                f"{conteos['sin_mascarilla']} sin mascarilla"
            )

        # B. Validación cuantitativa de faltantes
        # RN3 no tiene clases negativas para casco ni chaleco.

        if conteos['casco'] < personas:
            faltas.append(
                f"{personas - conteos['casco']} casco(s) faltante(s)"
            )

        if conteos['chaleco'] < personas:
            faltas.append(
                f"{personas - conteos['chaleco']} chaleco(s) faltante(s)"
            )

        # Evita duplicar la falta de guantes si ya se detectó sin_guantes.
        if conteos['guantes'] < (personas * 1) and conteos['sin_guantes'] == 0:
            faltas.append("Falta de guantes de protección")

        # CASO B: Sin novedades (EPP completo)
        if len(faltas) == 0:
            return jsonify({
                "estado": "Procesado",
                "infraccion": False,
                "motivo": "EPP completo y verificado",
                "docente_notificado": False
            }), 200

        # CASO C: REGISTRO DE INFRACCIÓN DETECTADA
        infraccion_detectada = True

        total_epp_positivo = sum([
            conteos[k]
            for k in [
                'casco',
                'guantes',
                'lentes',
                'chaleco',
                'mascarilla'
            ]
        ])

        # Determinación automática de la severidad del incidente
        if total_epp_positivo == 0:
            severidad = "ALTA"
            descripcion = (
                "Incumplimiento crítico: Personal detectado sin "
                "equipo de protección en zona de riesgo."
            )
        else:
            severidad = "MEDIA"
            descripcion = f"Faltante detectado: {', '.join(faltas)}."

        print("\n========== ALERTA INFRACCIÓN S.E.P.P. ==========")
        print("Cámara:", codigo_camara)
        print("Severidad:", severidad)
        print("Detalle:", descripcion)
        print("================================================\n")

        # Generación y subida del enlace de evidencia fotográfica
        url_evidencia = subir_a_imgbb(ruta_temporal)

        if id_docente not in (None, "", 0, "0"):
            id_docente = int(id_docente)

            # Persistencia de la alerta en PostgreSQL
            cur.execute("""
                INSERT INTO alertas (
                    codigo_camara,
                    id_docente,
                    tipo_falta,
                    severidad,
                    fecha_hora,
                    evidencia_url
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                codigo_camara,
                id_docente,
                descripcion,
                severidad,
                datetime.now(),
                url_evidencia
            ))

            conn.commit()
            print("Alerta guardada exitosamente en Neon.tech")

        # Despacho de alerta hacia el servidor Node.js
        try:
            respuesta_node = requests.post(
                NODE_URL,
                json={
                    "codigo_camara": codigo_camara,
                    "tipo_falta": descripcion,
                    "severidad": severidad,
                    "evidencia_url": url_evidencia,
                    "id_docente": id_docente
                },
                timeout=4
            )

            respuesta_node.raise_for_status()
            print("Notificación enviada a Node.js correctamente")

        except Exception as e:
            print("Error notificando al servicio Node.js:", e)

        return jsonify({
            "estado": "Procesado",
            "infraccion": True,
            "motivo": descripcion,
            "docente_notificado": bool(id_docente)
        }), 200

    except Exception as e:
        print("========== ERROR CRÍTICO EN S.E.P.P. ==========")
        traceback.print_exc()
        print("===============================================")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cur:
                cur.close()
            if conn:
                conn.close()
            if ruta_temporal and os.path.exists(ruta_temporal):
                os.remove(ruta_temporal)
        except Exception:
            traceback.print_exc()

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=True
    )