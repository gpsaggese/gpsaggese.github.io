"""
Flask Backend API for PEFT Fake News Detector

Loads the fine-tuned RoBERTa model with LoRA adapters and provides
REST API endpoints for predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the PEFT model and tokenizer."""
    global model, tokenizer
    
    try:
        # Path to the checkpoint - use absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_dir, "../Code_files/roberta_lora_results/checkpoint-800")
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        print(f"Looking for model at: {checkpoint_path}")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            
            # Load PEFT config and model
            config = PeftConfig.from_pretrained(checkpoint_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                num_labels=2
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            
            print("✅ Model loaded successfully from checkpoint")
        else:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}")
            print("Loading base RoBERTa model without LoRA adapters...")
            
            model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2
            )
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            
            print("✅ Base model loaded (no fine-tuning)")
        
        model.to(device)
        model.eval()
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def predict_sentiment(text):
    """
    Predict if the news article is fake or true.
    
    Args:
        text (str): News article text
        
    Returns:
        dict: Prediction results with probabilities
    """
    if model is None or tokenizer is None:
        return None, "Model not loaded"
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        # Extract probabilities (Label 0 = True News, Label 1 = Fake News)
        true_prob = probabilities[0][0].item() * 100  # Label 0 = True
        fake_prob = probabilities[0][1].item() * 100  # Label 1 = Fake
        
        # Determine prediction
        prediction = "Fake News" if predicted_class == 1 else "True News"
        confidence = max(true_prob, fake_prob)
        
        # Generate warning message
        if confidence >= 90:
            warning = "High confidence prediction"
        elif confidence >= 70:
            warning = "Moderate confidence prediction"
        else:
            warning = "Low confidence - results may be uncertain"
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "true_probability": round(true_prob, 2),
            "fake_probability": round(fake_prob, 2),
            "warning": warning
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# API Routes

@app.route('/')
def home():
    """Root endpoint with API information."""
    return jsonify({
        "service": "PEFT Fake News Detector API",
        "version": "1.0",
        "model": "RoBERTa-base + LoRA",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict (POST)"
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expects JSON: {"text": "news article text"}
    Returns JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request"
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) < 10:
            return jsonify({
                "error": "Text is too short. Please provide at least 10 characters."
            }), 400
        
        # Get prediction
        result, error = predict_sentiment(text)
        
        if error:
            return jsonify({"error": error}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting PEFT Fake News Detector Backend API")
    print("=" * 60)
    
    # Load model on startup
    print("\n📦 Loading model...")
    success = load_model()
    
    if not success:
        print("\n⚠️  WARNING: Model failed to load. API will return errors.")
        print("Please check that the model checkpoint exists.")
    
    print("\n" + "=" * 60)
    print("✅ Server starting on http://localhost:5001")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /              - API information")
    print("  GET  /api/health    - Health check")
    print("  POST /api/predict   - Predict fake/true news")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Start Flask server (using port 5001 to avoid macOS AirPlay Receiver on 5000)
    app.run(host='0.0.0.0', port=5001, debug=False)
