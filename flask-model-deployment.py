import joblib
from flask import Flask, request, jsonify

model = 'houseprediction.py'

joblib.dump(model, "houseprediction.pkl")

app = Flask(__name__)
app = Flask("House Prediction App")

loaded_model = joblib.load('houseprediction.pkl')


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]  # Replace with your input data format

        # Make predictions using the loaded model
        predictions = model.predict(features)

        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
