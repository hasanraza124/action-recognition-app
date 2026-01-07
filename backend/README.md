# Backend Server for Action Recognition Model

This backend server serves the trained Keras model (`best_model.h5`) via a REST API.

## Setup Instructions

### 1. Create a Virtual Environment (Recommended)

```bash
# Navigate to the backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Model

The server expects the model file at `../download/best_model.h5`. Make sure:
- The model file exists at the correct location
- Update `CLASS_NAMES` in `app.py` to match your model's output classes

### 4. Run the Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

## API Endpoints

### POST /predict
Upload an image to get action recognition predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file with key "image"

**Response:**
```json
{
  "success": true,
  "prediction": "dancing",
  "confidence": 0.95,
  "caption": "Person is dancing (confidence: 95.00%)",
  "actions": [
    {
      "action": "dancing",
      "confidence": 0.95
    },
    {
      "action": "jumping",
      "confidence": 0.03
    },
    {
      "action": "running",
      "confidence": 0.01
    }
  ]
}
```

### GET /health
Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET /
Get API information.

## Important Notes

1. **Model Input Shape**: The preprocessing function automatically detects the model's expected input shape. If your model requires specific preprocessing (e.g., different normalization), update the `preprocess_image()` function.

2. **Class Names**: Update the `CLASS_NAMES` list in `app.py` to match the classes your model was trained on.

3. **CORS**: CORS is enabled for all origins. In production, you should restrict this to your frontend domain.

4. **Image Formats**: The server accepts common image formats (JPEG, PNG, etc.) and automatically converts them to RGB.

## Troubleshooting

- **Model Loading Error**: Ensure the model path is correct and the model file is not corrupted
- **Prediction Error**: Check that image preprocessing matches your model's training preprocessing
- **Port Already in Use**: Change the port in `app.py` if 5000 is already in use
