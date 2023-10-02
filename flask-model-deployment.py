


import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
app = Flask("House Prediction App")


# Load the saved model
model = joblib.load("your_saved_model.pkl")

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
