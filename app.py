from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_cardio.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "API de predicción lista."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "JSON vacío"}), 400

        features = body.get("features") or body.get("valores") or body.get("values")
        if features is None:
            return jsonify({"error": "Envía JSON con la clave 'features'"}), 400

        arr = np.array(features).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred = modelo.predict(arr_scaled)[0]
        return jsonify({"prediccion": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
