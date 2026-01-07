from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'download', 'best_model.h5')
model = None

# Class names or action labels (customize based on your model)
# Update these based on what your model was trained to predict
CLASS_NAMES = [
    'calling',
    'clapping',
    'cycling',
    'dancing',
    'drinking',
    'eating',
    'fighting',
    'hugging',
    'laughing',
    'listening_to_music',
    'running',
    'sitting',
    'sleeping',
    'texting',
    'using_laptop'
]

def load_model():
    """Load the Keras model"""
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        
        # Register custom objects - the model uses NotEqual layer
        custom_objects = {
            'NotEqual': keras.layers.Lambda(lambda x: tf.not_equal(x[0], x[1])),
        }
        
        # Load model without compiling (avoids optimizer issues)
        model = keras.models.load_model(
            MODEL_PATH, 
            compile=False,
            custom_objects=custom_objects
        )
        
        print(f"✓ Model loaded successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease ensure:")
        print("  1. The model file exists at: " + MODEL_PATH)
        print("  2. The model was saved with a compatible TensorFlow/Keras version")
        print("  3. If the model uses custom layers, they need to be registered")
        raise

def preprocess_image(image_file):
    """
    Preprocess the image for model prediction
    Adjust this based on your model's requirements
    """
    # Open the image
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get the expected input shape from the model
    # Assuming shape is (None, height, width, channels)
    input_shape = model.input_shape
    target_size = (input_shape[1], input_shape[2])
    
    # Resize the image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values (adjust based on your model's training)
    # Common normalization: divide by 255 to get values between 0 and 1
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_action_caption(predicted_class, confidence):
    """
    Generate a caption based on the predicted action
    """
    captions = {
        'calling': f"Person is making a phone call (confidence: {confidence:.2%})",
        'clapping': f"Person is clapping hands (confidence: {confidence:.2%})",
        'cycling': f"Person is riding a bicycle (confidence: {confidence:.2%})",
        'dancing': f"Person is dancing (confidence: {confidence:.2%})",
        'drinking': f"Person is drinking (confidence: {confidence:.2%})",
        'eating': f"Person is eating food (confidence: {confidence:.2%})",
        'fighting': f"Person is in a fighting pose (confidence: {confidence:.2%})",
        'hugging': f"Person is hugging (confidence: {confidence:.2%})",
        'laughing': f"Person is laughing (confidence: {confidence:.2%})",
        'listening_to_music': f"Person is listening to music (confidence: {confidence:.2%})",
        'running': f"Person is running (confidence: {confidence:.2%})",
        'sitting': f"Person is sitting (confidence: {confidence:.2%})",
        'sleeping': f"Person is sleeping (confidence: {confidence:.2%})",
        'texting': f"Person is texting on phone (confidence: {confidence:.2%})",
        'using_laptop': f"Person is using a laptop (confidence: {confidence:.2%})"
    }
    
    return captions.get(predicted_class, f"Action detected: {predicted_class} (confidence: {confidence:.2%})")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive an image and return predictions
    """
    try:
        # Check if image is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Preprocess the image
        processed_image = preprocess_image(image_file)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        if predicted_class_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_actions = []
        for idx in top_3_indices:
            if idx < len(CLASS_NAMES):
                action_name = CLASS_NAMES[idx]
            else:
                action_name = f"Class_{idx}"
            
            top_3_actions.append({
                'action': action_name,
                'confidence': float(predictions[0][idx])
            })
        
        # Generate caption
        caption = get_action_caption(predicted_class, confidence)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'caption': caption,
            'actions': top_3_actions
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Action Recognition API',
        'endpoints': {
            '/predict': 'POST - Upload image for prediction',
            '/health': 'GET - Check API health'
        }
    })

if __name__ == '__main__':
    # Load the model when the app starts
    load_model()
    
    # Run the Flask app
    print("Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
