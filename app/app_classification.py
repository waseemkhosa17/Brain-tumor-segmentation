import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import torchvision.transforms as transforms

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Set matplotlib backend to Agg to avoid thread issues
import matplotlib
matplotlib.use('Agg')  # Important: Add this before importing pyplot
import matplotlib.pyplot as plt

try:
    from src.model_classification import get_model
    print("Successfully imported classification modules")
except ImportError as e:
    print(f"Import error: {e}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'brain-tumor-classification-secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs/predictions', exist_ok=True)

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def load_model():
    """Load the trained classification model"""
    global model
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    
    try:
        model = get_model(model_name='resnet18', num_classes=4, pretrained=False)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Loaded trained classification model successfully!")
        else:
            print("Using untrained model for demo purposes.")
        
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using random model for demo")

def predict_image(image):
    """Predict tumor class from image"""
    try:
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities and convert to Python float
            all_probs = probabilities.cpu().numpy()[0]
            all_probs = [float(prob) for prob in all_probs]  # Convert numpy to Python float
            
        return class_names[predicted.item()], float(confidence.item()), all_probs
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown", 0.0, [0.25, 0.25, 0.25, 0.25]

def create_result_image(image, prediction, confidence, all_probs):
    """Create visualization of prediction"""
    # Create figure without using Tkinter backend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show original image
    ax1.imshow(image)
    ax1.set_title('Uploaded MRI Image')
    ax1.axis('off')
    
    # Show prediction probabilities
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = ax2.barh(class_names, all_probs, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_title('Tumor Classification Probabilities')
    ax2.set_xlabel('Probability')
    
    # Add probability values on bars
    for i, v in enumerate(all_probs):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Explicitly close the figure
    
    return img_base64

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index_classification.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Read and convert image
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            
            # Run prediction
            prediction, confidence, all_probs = predict_image(image)
            
            # Create visualization
            result_image = create_result_image(image, prediction, confidence, all_probs)
            
            # Create detailed probability list (ensure all are Python floats)
            prob_details = []
            for i, class_name in enumerate(class_names):
                prob_details.append({
                    'class': class_name,
                    'probability': round(float(all_probs[i]) * 100, 2)  # Convert to float
                })
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'confidence': round(float(confidence) * 100, 2),  # Convert to float
                'image': f'data:image/png;base64,{result_image}',
                'probabilities': prob_details
            })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sample')
def sample():
    """Return a sample prediction with demo data"""
    try:
        # Create a sample image for demo
        image = Image.new('RGB', (224, 224), color='gray')
        
        # Demo prediction (random for demo)
        demo_probs = np.random.dirichlet(np.ones(4), size=1)[0]
        demo_pred = np.argmax(demo_probs)
        
        prediction = class_names[demo_pred]
        confidence = float(demo_probs[demo_pred])  # Convert to float
        
        # Convert all probabilities to Python floats
        demo_probs = [float(prob) for prob in demo_probs]
        
        # Create visualization
        result_image = create_result_image(image, prediction, confidence, demo_probs)
        
        # Create probability details
        prob_details = []
        for i, class_name in enumerate(class_names):
            prob_details.append({
                'class': class_name,
                'probability': round(demo_probs[i] * 100, 2)
            })
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'image': f'data:image/png;base64,{result_image}',
            'probabilities': prob_details,
            'note': 'This is a sample demo with random data'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load model when app starts
with app.app_context():
    load_model()

if __name__ == '__main__':
    print("Starting Brain Tumor Classification Web App...")
    print(f"Using device: {device}")
    app.run(debug=True, host='0.0.0.0', port=5000)