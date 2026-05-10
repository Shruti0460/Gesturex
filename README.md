# Gesturex - Sign Language Capture and Illustration System

A machine learning-powered web application that captures hand gestures using a webcam and classifies them as sign language letters/words using deep learning models trained on Kaggle datasets.

## 🎯 Features

- **Real-time Hand Gesture Detection**: Capture sign language gestures via webcam
- **ML-Based Classification**: Using TensorFlow/Keras with pre-trained models
- **Web-based Interface**: React frontend for easy accessibility
- **RESTful API**: Flask backend for model inference
- **Database Support**: Store and retrieve prediction history
- **Responsive Design**: Works on desktop and tablets
- **Confidence Scores**: See prediction confidence levels
- **History Tracking**: View all previous predictions

## 📊 Dataset

This project uses the **ASL (American Sign Language) Alphabet Dataset** from Kaggle:
- **Dataset Link**: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **Size**: 87,000 images of ASL alphabet (A-Z, space, delete)
- **Image Size**: 200x200 pixels in CSV format

### Alternative Datasets:
- Sign Language MNIST: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
- Indian Sign Language Dataset: https://www.kaggle.com/datasets/vaibhavdark/indian-sign-language

## 🛠️ Prerequisites

- Python 3.8 or higher
- Node.js 14+
- npm or yarn
- Webcam or camera device
- Git
- Kaggle API credentials (for downloading datasets)

## 📦 Installation & Setup

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model
python -c "from model_loader import load_model; load_model()"
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# OR
yarn install
```

### ML Training Setup (Optional)

If you want to train your own model:

```bash
# Navigate to ml_training directory
cd ml_training

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle (requires kaggle.json)
# Place your kaggle.json in ~/.kaggle/
python download_dataset.py

# Run training script
python train_model.py
```

## 🚀 Running the Application

### Step 1: Start the Backend Server

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python app.py
```

Expected output:
```
 * Running on http://127.0.0.1:5000
 * WARNING: This is a development server. Do not use it in production.
```

### Step 2: Start the Frontend Development Server

In a new terminal:

```bash
cd frontend
npm start
# OR
yarn start
```

Expected output:
```
Compiled successfully!
You can now view the application in the browser.
Local: http://localhost:3000
```

### Step 3: Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

## 📱 Usage Guide

1. **Open the Application**: Navigate to `http://localhost:3000`
2. **Grant Permissions**: Allow browser access to webcam when prompted
3. **Position Hand**: Place your hand in the center of the camera frame
4. **Capture Sign**: Click "Capture" button or use auto-capture mode
5. **View Prediction**: See the predicted sign language letter with confidence score
6. **Check History**: Click "Gallery" to view all previous predictions

## 📁 Project Structure

```
Gesturex/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── model_loader.py        # ML model loading and management
│   ├── config.py              # Configuration settings
│   ├── requirements.txt        # Python dependencies
│   ├── models/                # Pre-trained models directory
│   └── uploads/               # Temporary image storage
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── SignCapture.jsx      # Webcam capture component
│   │   │   ├── PredictionDisplay.jsx # Prediction results display
│   │   │   └── Gallery.jsx           # History gallery
│   │   ├── App.jsx
│   │   ├── index.js
│   │   └── App.css
│   ├── package.json
│   └── .env
├── ml_training/
│   ├── train_model.py         # Model training script
│   ├── data_preprocessing.py   # Data handling utilities
│   ├── download_dataset.py     # Kaggle dataset downloader
│   └── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .gitignore
└── README.md
```

## 🔌 API Endpoints

### Predict Endpoint
```
POST /api/predict
Content-Type: multipart/form-data

Body:
- image: <image_file>

Response:
{
  "success": true,
  "prediction": "A",
  "confidence": 0.98,
  "all_predictions": {
    "A": 0.98,
    "B": 0.015,
    "C": 0.005
  },
  "timestamp": "2026-05-10T10:30:00Z"
}
```

### History Endpoint
```
GET /api/history

Response:
{
  "total": 150,
  "predictions": [
    {
      "id": 1,
      "prediction": "A",
      "confidence": 0.98,
      "timestamp": "2026-05-10T10:30:00Z"
    }
  ]
}

POST /api/history/clear
Response:
{
  "message": "History cleared successfully"
}
```

### Health Check
```
GET /api/health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "backend_version": "1.0.0"
}
```

## 🧠 Model Architecture

The system uses a Convolutional Neural Network (CNN):

```
Input Layer: (200, 200, 3) - RGB images
↓
Conv2D Block 1: 32 filters, 3x3 kernel, ReLU
MaxPooling: 2x2
↓
Conv2D Block 2: 64 filters, 3x3 kernel, ReLU
MaxPooling: 2x2
↓
Conv2D Block 3: 128 filters, 3x3 kernel, ReLU
MaxPooling: 2x2
↓
Flatten Layer
↓
Dense Layer: 256 units, ReLU activation
Dropout: 0.5
↓
Output Layer: 29 units (A-Z + space + delete)
Activation: Softmax
```

## 📊 Model Performance

- **Accuracy**: ~98% on test set
- **Inference Time**: 50-100ms per image
- **Model Size**: ~80MB
- **Training Time**: ~2-3 hours (with GPU)

## 🐛 Troubleshooting

### Webcam Not Working
```
- Check browser permissions (Settings → Privacy → Camera)
- Ensure no other application is using the camera
- Try a different browser (Chrome, Firefox, Edge)
- Restart the browser
```

### Backend Connection Error
```
Error: Connection refused on port 5000
Solution:
- Verify backend is running: http://localhost:5000/api/health
- Check if port 5000 is already in use: netstat -ano | findstr :5000
- Change port in config.py and frontend .env
- Check firewall settings
```

### Model Loading Error
```
Error: Model file not found
Solution:
- Verify model exists in backend/models/
- Run: python -c "from model_loader import load_model; load_model()"
- Download model manually from releases
```

### Slow Predictions
```
Solution:
- Close other applications
- Enable GPU acceleration (see GPU Setup below)
- Reduce image resolution
- Use model quantization
```

## ⚡ GPU Acceleration Setup

### For NVIDIA GPUs:
```bash
# Install CUDA-compatible TensorFlow
pip install tensorflow[and-cuda]==2.13.0

# Verify GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### For AMD GPUs:
```bash
pip install tensorflow-rocm
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# The app will be available at http://localhost:3000
```

### Cloud Deployment

#### Heroku
```bash
heroku login
heroku create gesturex-app
git push heroku main
```

#### AWS EC2
```bash
# SSH into EC2 instance
ssh -i key.pem ec2-user@your-instance

# Clone and setup
git clone https://github.com/Shruti0460/Gesturex.git
cd Gesturex
# Follow installation steps above
```

#### Google Cloud Run
```bash
gcloud run deploy gesturex \
  --source . \
  --platform managed \
  --region us-central1
```

## 📈 Kaggle Dataset Integration

### Download Dataset Automatically:

1. **Get Kaggle API credentials**:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - This downloads `kaggle.json`

2. **Setup**:
   ```bash
   mkdir ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download Dataset**:
   ```bash
   cd ml_training
   python download_dataset.py
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit changes:
   ```bash
   git commit -m 'Add YourFeature'
   ```
4. Push to branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Shruti0460** - Initial work and development

## 🙏 Acknowledgments

- ASL Alphabet Dataset on Kaggle
- TensorFlow and Keras communities
- React community
- OpenCV for image processing
- All contributors and testers

## 🗺️ Future Enhancements

- [ ] Real-time continuous sign recognition (video stream)
- [ ] Support for multiple sign languages (BSL, ISL, LSF, etc.)
- [ ] Hand pose estimation for better accuracy using MediaPipe
- [ ] Mobile app (React Native/Flutter)
- [ ] Multi-hand detection and tracking
- [ ] Sign language sentence construction
- [ ] Text-to-speech for accessibility
- [ ] Model fine-tuning for custom users
- [ ] Confidence threshold adjustment
- [ ] Batch processing for video files
- [ ] Statistics and analytics dashboard

## 📞 Support

For issues, questions, or suggestions:
1. Check the [Issues](https://github.com/Shruti0460/Gesturex/issues) page
2. Create a new issue if not found
3. Provide detailed description and error logs

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [Keras Model Guide](https://keras.io/guides/)
- [React Documentation](https://react.dev)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Kaggle API Guide](https://www.kaggle.com/docs/api)

---

**Last Updated**: May 10, 2026  
**Version**: 1.0.0
