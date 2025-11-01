import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import io

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Set the device: use CUDA if available otherwise CPU.
device = torch.device("CUDA" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model(model_path):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Linear(model._fc.in_features, 2)
    state_dict = torch.load(model_path, map_location=device)
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Set up transform
val_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Load pretrained weights
MODEL_PATH = 'hcquery_model.pt'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    model.to(device)
else:
    print(f"Warning: Model file {MODEL_PATH} not found!")

def predict_image(model, image_path):
    """
    Perform inference on a single image
    """
    try:
        image = Image.open(image_path).convert("RGB") 
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            class_1_probability = probabilities[:, 1].item()  # Probability for the positive class
        return class_1_probability, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        probability, error = predict_image(model, filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if error:
            return jsonify({'error': f'Prediction failed: {error}'}), 500
        
        # Convert probability to percentage
        lrs_score = probability * 100
        
        return jsonify({
            'success': True,
            'lrs_score': round(lrs_score, 2),
            'probability': round(probability, 4),
            'interpretation': get_interpretation(lrs_score)
        })
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or TIFF'}), 400

def get_interpretation(score):
    """Provide interpretation of the LRS score"""
    if score < 20:
        return "Low likelihood of retinopathy"
    elif score < 50:
        return "Moderate likelihood of retinopathy"
    elif score < 80:
        return "High likelihood of retinopathy"
    else:
        return "Very high likelihood of retinopathy"

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)