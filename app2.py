import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import json
from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)

# Load the SAME CNN architecture
class SimplePestCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(SimplePestCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load pest info
with open('data/pest_info.json', 'r') as f:
    PEST_INFO = json.load(f)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("pest_classifier2.pth", map_location=device)

# Initialize model with correct number of classes
model = SimplePestCNN(num_classes=len(checkpoint['classes']))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Get class mappings
classes = checkpoint['classes']
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

print(f"✓ Loaded custom CNN model")
print(f"✓ Best accuracy: {checkpoint['best_acc']:.2%}")
print(f"✓ Classes: {classes}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    temp_path = 'temp_image.jpg'
    file.save(temp_path)
    
    # Preprocess and predict
    image = Image.open(temp_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    pest_idx = predicted.item()
    pest_type = idx_to_class[pest_idx]
    confidence_score = confidence.item()
    
    treatments = PEST_INFO.get(pest_type, ["Check pest_info.json"])
    
    os.remove(temp_path)
    
    return jsonify({
        'pest': pest_type,
        'confidence': f'{confidence_score:.2%}',
        'treatments': treatments
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)