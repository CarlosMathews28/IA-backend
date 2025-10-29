from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # permite llamadas desde tu frontend (Netlify/Bolt)

# Rutas absolutas a los artefactos
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_cardio.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Cargar modelo y scaler al iniciar
modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "API de predicción lista."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "JSON vacío"}), 400

        # Aceptamos 'features' | 'valores' | 'values'
        features = body.get("features") or body.get("valores") or body.get("values")
        if features is None:
            return jsonify({"error": "Envía JSON con la clave 'features' (lista o lista de listas)"}), 400

        # Normalizar la forma de entrada:
        # - Si viene una sola fila [x1, x2, ...] -> convertir a [[x1, x2, ...]]
        # - Si viene lote [[...], [...]] -> dejar igual
        arr = np.asarray(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            return jsonify({"error": "El formato de 'features' debe ser 1D o 2D"}), 400

        # Escalar y predecir
        arr_scaled = scaler.transform(arr)
        preds = modelo.predict(arr_scaled).astype(int).tolist()

        # (Opcional) Probabilidades si el modelo las soporta
        prob = None
        if hasattr(modelo, "predict_proba"):
            prob = modelo.predict_proba(arr_scaled).tolist()

        return jsonify({"predictions": preds, "probabilities": prob})

    except Exception as e:
        # Para depurar rápido en Render podrías loguear e
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Puerto dinámico para Render; 5000 en local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
