from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import warnings
import pickle
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'download', 'best_model.h5')
VOCAB_PATH = os.path.join(os.path.dirname(__file__), '..', 'download', 'tokenizer.pkl')  # If you have this
model = None
tokenizer = None
max_length = 35  # Typical max caption length

# Start and end tokens for caption generation
START_TOKEN = '<start>'
END_TOKEN = '<end>'

# Custom layer for NotEqual operation
class NotEqualLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqualLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        if isinstance(inputs, list):
            return tf.not_equal(inputs[0], inputs[1])
        return inputs
    
    def get_config(self):
        return super(NotEqualLayer, self).get_config()

def load_tokenizer():
    """Load tokenizer if available"""
    global tokenizer
    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"✓ Tokenizer loaded from {VOCAB_PATH}")
        except:
            print(f"⚠ Could not load tokenizer from {VOCAB_PATH}")
    else:
        print(f"⚠ Tokenizer file not found at {VOCAB_PATH}")
        print(f"  The model will return raw predictions")

def load_model():
    """Load the Keras model"""
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        
        # Register custom objects - the model uses NotEqual layer
        custom_objects = {
            'NotEqual': NotEqualLayer,
        }
        
        # Load model without compiling (avoids optimizer issues)
        model = keras.models.load_model(
            MODEL_PATH, 
            compile=False,
            custom_objects=custom_objects
        )
        
        print(f"✓ Model loaded successfully!")
        print(f"  Model has {len(model.inputs)} inputs")
        for i, inp in enumerate(model.inputs):
            print(f"    Input {i}: {inp.name} - Shape: {inp.shape}")
        print(f"  Output shape: {model.output_shape}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def extract_image_features(image_file):
    """
    Extract features from image using a pre-trained model (e.g., InceptionV3 or ResNet)
    """
    # Load and preprocess image
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to expected size (typically 224x224 or 299x299)
    target_size = (224, 224)  # Adjust if your model uses different size
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use pre-trained model to extract features (2048-dim vector typically)
    # For now, return a placeholder - you may need to use InceptionV3 or ResNet
    feature_extractor = keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Preprocess for InceptionV3
    img_for_inception = image.resize((299, 299))
    img_array = np.array(img_for_inception)
    img_array = keras.applications.inception_v3.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    features = feature_extractor.predict(img_array, verbose=0)
    return features

def generate_caption(image_features):
    """
    Generate caption from image features using the loaded model
    """
    # Start with <start> token
    caption = [START_TOKEN]
    
    for _ in range(max_length):
        # Convert caption to sequence
        if tokenizer:
            sequence = tokenizer.texts_to_sequences([' '.join(caption)])[0]
        else:
            # If no tokenizer, use dummy sequence
            sequence = list(range(len(caption)))
        
        # Pad sequence
        sequence = keras.preprocessing.sequence.pad_sequences(
            [sequence], 
            maxlen=max_length, 
            padding='post'
        )
        
        # Predict next word
        predictions = model.predict([image_features, sequence], verbose=0)
        predicted_id = np.argmax(predictions[0])
        
        # Convert ID to word
        if tokenizer and hasattr(tokenizer, 'index_word'):
            predicted_word = tokenizer.index_word.get(predicted_id, END_TOKEN)
        else:
            predicted_word = f"word_{predicted_id}"
        
        if predicted_word == END_TOKEN or predicted_word == 'end':
            break
        
        caption.append(predicted_word)
    
    # Remove <start> token and join
    caption_text = ' '.join([word for word in caption if word not in [START_TOKEN, 'start']])
    return caption_text

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive an image and return caption/prediction
    """
    try:
        # Check if image is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"Processing image: {image_file.filename}")
        
        # Extract image features
        image_features = extract_image_features(image_file)
        
        # Generate caption
        caption = generate_caption(image_features)
        
        print(f"Generated caption: {caption}")
        
        # Prepare response for the frontend
        response = {
            'success': True,
            'caption': caption.capitalize() if caption else "Unable to generate caption",
            'actions': [
                {
                    'action': 'Caption Generated',
                    'confidence': 0.95
                },
                {
                    'action': 'Image Processed',
                    'confidence': 1.0
                }
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'caption': 'Error processing image',
            'actions': []
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Image Captioning API',
        'model_info': 'LSTM-based image captioning model',
        'endpoints': {
            '/predict': 'POST - Upload image for caption generation',
            '/health': 'GET - Check API health'
        }
    })

if __name__ == '__main__':
    # Load the model and tokenizer when the app starts
    load_tokenizer()
    load_model()
    
    # Run the Flask app
    print("\n" + "="*60)
    print("Starting Flask server on http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
