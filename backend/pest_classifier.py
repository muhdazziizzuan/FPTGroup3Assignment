#!/usr/bin/env python3
"""
Pest Classification Script
Integrates with the trained CNN model for real pest identification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import json
import argparse
import os
import sys

# Define depthwise separable convolution block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

# Define the CNN model architecture (should match your trained model)
class PestClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(PestClassifier, self).__init__()
        # Custom MobileNetV2-like features to match the saved model structure
        self.features = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # features.0
            nn.BatchNorm2d(32),  # features.1
            nn.ReLU6(inplace=True),
            
            # Depthwise separable conv blocks (features.3 to features.15)
            DepthwiseSeparableConv(32, 64, 1),    # features.3
            DepthwiseSeparableConv(64, 128, 2),   # features.4
            DepthwiseSeparableConv(128, 128, 1),  # features.5
            DepthwiseSeparableConv(128, 256, 2),  # features.6
            DepthwiseSeparableConv(256, 256, 1),  # features.7
            DepthwiseSeparableConv(256, 512, 2),  # features.8
            DepthwiseSeparableConv(512, 512, 1),  # features.9
            DepthwiseSeparableConv(512, 512, 1),  # features.10
            DepthwiseSeparableConv(512, 512, 1),  # features.11
            DepthwiseSeparableConv(512, 512, 1),  # features.12
            DepthwiseSeparableConv(512, 512, 1),  # features.13
            DepthwiseSeparableConv(512, 1024, 2), # features.14
            DepthwiseSeparableConv(1024, 1024, 1) # features.15
        )
        
        # Simple linear classifier to match the saved model
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ResNet50-based classifier for the ResNet50 model
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=12):
        super(ResNet50Classifier, self).__init__()
        # Use ResNet50 architecture without pretrained weights
        resnet = models.resnet50(weights=None)
        
        # Copy all layers except the final fc layer
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Custom classifier to match the saved model structure
        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Class names mapping
CLASS_NAMES = [
    'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
    'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
]

# Pest descriptions and treatment suggestions
PEST_INFO = {
    'ants': {
        'description': 'Common garden ants that can damage plant roots and farm aphids.',
        'treatments': [
            'Use diatomaceous earth around affected areas',
            'Apply cinnamon or coffee grounds as natural deterrents',
            'Remove food sources and moisture'
        ]
    },
    'bees': {
        'description': 'Beneficial pollinators, generally not considered pests.',
        'treatments': [
            'Protect and encourage bee populations',
            'Plant bee-friendly flowers nearby',
            'Avoid pesticide use during flowering'
        ]
    },
    'beetle': {
        'description': 'Various beetle species that can damage leaves and crops.',
        'treatments': [
            'Hand-pick beetles in early morning',
            'Use row covers during peak season',
            'Apply neem oil spray'
        ]
    },
    'catterpillar': {
        'description': 'Larval stage of moths and butterflies, can cause significant leaf damage.',
        'treatments': [
            'Inspect plants regularly for eggs',
            'Use Bacillus thuringiensis (Bt) spray',
            'Encourage beneficial predators'
        ]
    },
    'earthworms': {
        'description': 'Beneficial soil organisms that improve soil health.',
        'treatments': [
            'Protect and encourage earthworm populations',
            'Add organic matter to soil',
            'Avoid chemical pesticides'
        ]
    },
    'earwig': {
        'description': 'Nocturnal insects that can damage seedlings and soft plant tissues.',
        'treatments': [
            'Remove hiding places like debris',
            'Use beer traps',
            'Apply diatomaceous earth'
        ]
    },
    'grasshopper': {
        'description': 'Jumping insects that can cause extensive damage to crops and gardens.',
        'treatments': [
            'Use row covers for protection',
            'Apply kaolin clay spray',
            'Encourage natural predators'
        ]
    },
    'moth': {
        'description': 'Adult stage of various species, some larvae can be crop pests.',
        'treatments': [
            'Use pheromone traps',
            'Apply beneficial nematodes',
            'Remove overwintering sites'
        ]
    },
    'slug': {
        'description': 'Soft-bodied mollusks that damage leaves and seedlings.',
        'treatments': [
            'Use copper barriers',
            'Apply iron phosphate baits',
            'Remove hiding places'
        ]
    },
    'snail': {
        'description': 'Shelled mollusks that feed on plant material.',
        'treatments': [
            'Hand-pick during evening hours',
            'Use beer traps',
            'Apply crushed eggshells around plants'
        ]
    },
    'wasp': {
        'description': 'Beneficial predators that control other pest insects.',
        'treatments': [
            'Protect beneficial wasp species',
            'Remove only if nesting in problematic areas',
            'Use natural deterrents if necessary'
        ]
    },
    'weevil': {
        'description': 'Beetle family that includes many serious crop pests.',
        'treatments': [
            'Use beneficial nematodes',
            'Apply diatomaceous earth',
            'Remove plant debris'
        ]
    }
}

def load_model(model_path):
    """Load the trained model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the checkpoint first to get class information
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Get class information from checkpoint if available
            if isinstance(checkpoint, dict) and 'classes' in checkpoint:
                classes = checkpoint['classes']
                num_classes = len(classes)
            else:
                # Fallback to default classes
                classes = CLASS_NAMES
                num_classes = len(CLASS_NAMES)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Determine model architecture based on the state dict keys
            state_keys = list(state_dict.keys())
            
            # Check if it's a ResNet50 model (has 'fc.fc.weight' or layer structure)
            is_resnet = any('layer1.' in key or 'layer2.' in key or 'layer3.' in key or 'layer4.' in key for key in state_keys)
            is_resnet = is_resnet or any('fc.fc.' in key for key in state_keys)
            
            if is_resnet:
                model = ResNet50Classifier(num_classes=num_classes)
                
                # For ResNet50, we need to handle the fc layer structure
                if any('fc.fc.' in key for key in state_keys):
                    # The saved model has fc.fc structure, but our model has fc.0
                    # We need to map the keys correctly
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key == 'fc.fc.weight':
                            new_state_dict['fc.0.weight'] = value
                        elif key == 'fc.fc.bias':
                            new_state_dict['fc.0.bias'] = value
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict
            else:
                model = PestClassifier(num_classes=num_classes)
            
            # Load the state dict
            model.load_state_dict(state_dict, strict=True)
                
            model.to(device)
            model.eval()
            return model, device, classes
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def preprocess_image(image_input):
    """Preprocess image for model input"""
    try:
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Handle both file paths and PIL Image objects
        if isinstance(image_input, str):
            # It's a file path
            image = Image.open(image_input).convert('RGB')
        elif hasattr(image_input, 'read'):
            # It's a file-like object
            image = Image.open(image_input).convert('RGB')
        else:
            # It's already a PIL Image
            image = image_input.convert('RGB')
            
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def classify_pest(image_input, model_path):
    """Classify pest in the given image"""
    try:
        # Load model
        model, device, classes = load_model(model_path)
        
        # Preprocess image
        image_tensor = preprocess_image(image_input)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = classes[predicted.item()]
            confidence_score = confidence.item()
            
            # Get pest information
            pest_info = PEST_INFO.get(predicted_class, {
                'description': 'Information not available for this pest.',
                'treatments': ['Consult with agricultural extension services']
            })
            
            result = {
                'pest_name': predicted_class,
                'confidence': confidence_score,
                'description': pest_info['description'],
                'treatment_suggestions': pest_info['treatments']
            }
            
            return result
            
    except Exception as e:
        return {
            'error': f"Classification failed: {str(e)}",
            'pest_name': None,
            'confidence': 0
        }

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 3:
        print("Usage: python pest_classifier.py <image_path> <model_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # Classify the pest
    result = classify_pest(image_path, model_path)
    
    # Output result as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()