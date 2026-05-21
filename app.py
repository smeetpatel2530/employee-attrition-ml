"""
Employee Attrition Prediction - Flask Web App
Author: Smeet Patel | M.Tech CSE, DTU
"""
from flask import Flask, request, jsonify, render_template
from src.predict import predict_attrition

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        result = predict_attrition(data)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "Employee Attrition Predictor v1.0"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
