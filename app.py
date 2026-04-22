from flask import Flask, request, jsonify, render_template
import joblib
import os
import re

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = 'static/model.pkl'
VECTORIZER_PATH = 'static/vectorizer.pkl'

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    model = None
    vectorizer = None

def preprocess(text):
    # Simple regex to simulate the minimal preprocessing
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded.'}), 500
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided.'}), 400
        
    text = data['text']
    clean_text = preprocess(text)
    
    # Vectorize
    vectorized_text = vectorizer.transform([clean_text])
    
    # Predict
    prediction = model.predict(vectorized_text)[0]
    
    # Probabilities (if applicable)
    confidence = 0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized_text)[0]
        confidence = float(max(probs))
    elif hasattr(model, "decision_function"):
        # For SVM, decision function output magnitude correlates with confidence loosely
        scores = model.decision_function(vectorized_text)[0]
        confidence = 1.0  # LinearSVC doesn't output true probabilities easily

    return jsonify({
        'sentiment': prediction,
        'confidence': confidence,
        'text': text
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
