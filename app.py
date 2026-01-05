from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)

# Load model at startup
MODEL_PATH = 'model/model.pkl'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {MODEL_PATH}")

# Mapping for categorical variables (for display purposes)
MAPPINGS = {
    'gender': {0: 'Female', 1: 'Male'},
    'occupation': {0: 'Accountant', 1: 'Artist', 2: 'Consultant', 3: 'Designer', 
                   4: 'Doctor', 5: 'Engineer', 6: 'Executive', 7: 'Lawyer', 
                   8: 'Manager', 9: 'Nurse', 10: 'Salesperson', 11: 'Student', 
                   12: 'Teacher', 13: 'Technician'},
    'education_level': {0: "Associate's", 1: "Bachelor's", 2: 'Doctoral', 
                        3: 'High School', 4: "Master's"},
    'marital_status': {0: 'Married', 1: 'Single'},
    'loan_status': {0: 'Approved', 1: 'Denied'}
}

# Reverse mappings for input
REVERSE_MAPPINGS = {
    key: {v: k for k, v in mapping.items()} 
    for key, mapping in MAPPINGS.items()
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get form data
        data = request.form
        
        # Create feature vector
        features = {
            'age': int(data['age']),
            'gender': REVERSE_MAPPINGS['gender'][data['gender']],
            'occupation': REVERSE_MAPPINGS['occupation'][data['occupation']],
            'education_level': REVERSE_MAPPINGS['education_level'][data['education_level']],
            'marital_status': REVERSE_MAPPINGS['marital_status'][data['marital_status']],
            'income': int(data['income']),
            'credit_score': int(data['credit_score'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get result
        result = MAPPINGS['loan_status'][prediction]
        confidence = max(prediction_proba) * 100
        
        return jsonify({
            'prediction': result,
            'confidence': f'{confidence:.2f}%',
            'probabilities': {
                'Approved': f'{prediction_proba[0] * 100:.2f}%',
                'Denied': f'{prediction_proba[1] * 100:.2f}%'
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)