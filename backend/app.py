from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import numpy as np
from PIL import Image
from model_loader import load_model, preprocess_image, predict
import config
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model on startup
try:
    model = load_model()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# In-memory storage for predictions (in production, use database)
predictions_history = []

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': config.APP_VERSION
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict_sign():
    """Predict sign language from uploaded image"""
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image provided'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No image selected'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Open and preprocess image
        image = Image.open(filepath)
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction, confidence, all_predictions = predict(model, processed_image)
        
        # Store in history
        prediction_record = {
            'id': len(predictions_history) + 1,
            'prediction': prediction,
            'confidence': float(confidence),
            'all_predictions': {str(k): float(v) for k, v in all_predictions.items()},
            'timestamp': datetime.now().isoformat(),
            'image_path': filepath
        }
        predictions_history.append(prediction_record)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'all_predictions': {str(k): float(v) for k, v in all_predictions.items()},
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    try:
        # Return without image paths for privacy
        history_data = [
            {
                'id': p['id'],
                'prediction': p['prediction'],
                'confidence': p['confidence'],
                'timestamp': p['timestamp']
            }
            for p in predictions_history
        ]
        
        return jsonify({
            'success': True,
            'total': len(predictions_history),
            'predictions': history_data[:100]  # Return last 100 predictions
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    global predictions_history
    try:
        predictions_history = []
        return jsonify({
            'success': True,
            'message': 'History cleared successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about predictions"""
    try:
        if not predictions_history:
            return jsonify({
                'total_predictions': 0,
                'unique_signs': 0,
                'average_confidence': 0
            }), 200
        
        predictions = [p['prediction'] for p in predictions_history]
        confidences = [p['confidence'] for p in predictions_history]
        
        stats = {
            'total_predictions': len(predictions_history),
            'unique_signs': len(set(predictions)),
            'average_confidence': np.mean(confidences),
            'most_common': max(set(predictions), key=predictions.count),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Gesturex Backend Server")
    print("="*50)
    print(f"Starting on {config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"Debug mode: {config.DEBUG_MODE}")
    print("="*50 + "\n")
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.DEBUG_MODE
    )
