from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Sabhi models load karna
models = {
    "model": pickle.load(open("model.pkl", "rb")),
    "logreg": pickle.load(open("logreg.pkl", "rb"))
}

@app.route('/')
def home():
    return "Welcome to Restaurant Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # User se model ka naam le lo (default: model)
    model_choice = data.get('model', 'model')
    
    if model_choice not in models:
        return jsonify({'error': 'Invalid model choice. Use model or logreg'})

    # Input features convert karo NumPy array me
    features = np.array(data['features']).reshape(1, -1)

    # Prediction karo
    prediction = models[model_choice].predict(features)

    return jsonify({'model_used': model_choice, 'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
