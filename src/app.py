from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
MODEL_DIR = './model/fin_sentiment_model'
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

# Ensure the model is in evaluation mode
model.eval()

# Preprocessing function
def preprocess_text(text):
    return text.lower()

# Predict function
def predict_sentiment(texts):
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Tokenize the input texts
    encodings = tokenizer(preprocessed_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1).numpy()  # Get predicted class indices
    print(outputs)
    return predictions

# Define the welcome page with file upload
@app.route('/')
def home():
    return render_template('index.html')

# Define the /predict endpoint to handle file uploads
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is provided
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if the file is a text file
    if not file.filename.endswith('.txt'):
        return jsonify({"error": "Only .txt files are supported"}), 400
    
    try:
        # Read the text file into a list of sentences
        content = file.read().decode('utf-8')
        texts = content.splitlines()  # Split into lines (one sentence per line)
        
        # Make predictions
        predictions = predict_sentiment(texts)
        
        # Map predictions back to labels (you can customize this based on your labels)
        label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        labeled_predictions = [label_mapping[pred] for pred in predictions]
        
        # Return predictions to the UI
        return render_template('results.html', results=zip(texts, labeled_predictions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_from_api():
    # Check if a file is provided
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if the file is a text file
    if not file.filename.endswith('.txt'):
        return jsonify({"error": "Only .txt files are supported"}), 400
    
    try:
        # Read the text file into a list of sentences
        content = file.read().decode('utf-8')
        texts = content.splitlines()  # Split into lines (one sentence per line)
        
        # Make predictions
        predictions = predict_sentiment(texts)
        
        # Map predictions back to labels
        label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        labeled_predictions = [label_mapping[pred] for pred in predictions]
        
        # Return predictions as JSON
        return jsonify({"predictions": [{"sentence": text, "sentiment": sentiment} for text, sentiment in zip(texts, labeled_predictions)]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
