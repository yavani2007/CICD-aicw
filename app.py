from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Extract and validate the features (expecting a list of 4 values)
        features = np.array(data["features"]).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(features)[0]
        
        # Map the numerical prediction to the class name
        classes = ["setosa", "versicolor", "virginica"]
        result = {"prediction": classes[prediction]}
        
        # Return the prediction as JSON
        return jsonify(result)
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)