import React from 'react';
import { Settings } from 'lucide-react';
import './ModelSelector.css';

const ModelSelector = ({ selectedModel, onModelChange, isAnalyzing }) => {
  const availableModels = [
    {
      id: 'pest_classifier2.pth',
      name: 'MobileNetV2 Classifier',
      description: 'Lightweight model optimized for mobile devices'
    },
    {
      id: 'best_resnet50_model.pth',
      name: 'ResNet50 Classifier',
      description: 'High-accuracy model with deeper architecture'
    }
  ];

  return (
    <div className="model-selector">
      <div className="model-selector-header">
        <Settings className="model-selector-icon" />
        <label htmlFor="model-select">Detection Model</label>
      </div>
      
      <select
        id="model-select"
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        disabled={isAnalyzing}
        className="model-select"
      >
        {availableModels.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
      
      <div className="model-description">
        {availableModels.find(m => m.id === selectedModel)?.description}
      </div>
    </div>
  );
};

export default ModelSelector;