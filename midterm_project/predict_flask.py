import pickle
import numpy as np
from flask import Flask, request, jsonify

# -------------------------------
# Load model
# -------------------------------
depth = 10
learning_rate = 0.02
iterations = 200
params_str = f"depth_{depth}_lr_{learning_rate}_iter_{iterations}"

model_file = 'model_depth_10_lr_0.02_iter_200.bin'

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

# -------------------------------
# Flask app
# -------------------------------
app = Flask("Mine")

@app.route("/predict_flask", methods=["POST"])
def predict():
    """Predict mine type and class probabilities for scanned area."""
    
    # input will be turned into a python dictionary:
    # ie A JSON payload sent via POST request body (e.g., {"Voltage": 0.24, "Height": 0.72, "soil_type_cat": 1})
    data = request.get_json()
    
    # DO NOT pass Mine_type from client — remove target variable if present
    if "Mine_type" in data:
        del data["Mine_type"]
    
    # Convert to vector
    X = dv.transform([data])
    
    # Predict probabilities
    proba = model.predict_proba(X)[0]          # array shape (n_classes,)
    predicted_class = int(np.argmax(proba)) + 1  # classes 1–5
    
    result = {"predicted_class": predicted_class,"class_probabilities": proba.tolist(),"confidence": float(max(proba)),}
    
    return jsonify(result) # python dict (ie results) returned as json
 

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9898)



