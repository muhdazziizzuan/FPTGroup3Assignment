#!/usr/bin/env python3
"""
Pest Classification Script
Integrates with the trained CNN model for real pest identification
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import sys
import os

# Define the CNN model architecture (should match your trained model)
class PestClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(PestClassifier, self).__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),  # Adjust based on input size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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
        model = PestClassifier(num_classes=len(CLASS_NAMES))
        
        # Load the trained weights
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, device
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def preprocess_image(image_path):
    """Preprocess the image for model input"""
    try:
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def classify_pest(image_path, model_path):
    """Classify pest in the given image"""
    try:
        # Load model
        model, device = load_model(model_path)
        
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = CLASS_NAMES[predicted.item()]
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