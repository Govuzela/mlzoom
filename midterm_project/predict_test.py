#!/usr/bin/env python3
import requests

url = 'http://localhost:9898/predict_flask'

mine = {
    'Voltage': 0.240966462513951,
    'Height': 0.727272727272727,
    'soil_type_cat': 1}

response = requests.post(url, json=mine).json()

print("API Response:", response)

mine_code = {1.0: 'No_mine',2.0: 'Anti_tank',3.0: 'Anti_personnel',4.0: 'Booby_trapped/Anti_personnel',5.0: 'M14_anti_personnel'}

try:
    predicted_numeric = response['predicted_class']
    predicted_label = mine_code.get(predicted_numeric, f"Unknown class {predicted_numeric}")
    
    print("Predicted mine class:", predicted_label)
    print(f"Confidence: {response['confidence']:.3f}")
    
    # Map list of probabilities to readable labels
    class_probs_list = response.get('class_probabilities', [])
    class_probabilities_readable = {
        mine_code.get(i + 1.0, f"Unknown class {i+1}"): prob
        for i, prob in enumerate(class_probs_list)
    }
    print("Class probabilities:", class_probabilities_readable)
    
except KeyError:
    print("Missing expected fields in response:", response)
