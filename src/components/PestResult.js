import React from 'react';
import { Bug, Target, AlertCircle, CheckCircle, RotateCcw } from 'lucide-react';

const PestResult = ({ result, onNewAnalysis }) => {
  if (!result) return null;

  const { pest_name, confidence, error, description, treatment_suggestions } = result;

  if (error) {
    return (
      <div className="result-container">
        <div className="result-header">
          <AlertCircle className="result-icon" style={{ color: '#d32f2f' }} />
          <h3 style={{ color: '#d32f2f' }}>Analysis Error</h3>
        </div>
        <div className="error-message">
          {error}
        </div>
        <button onClick={onNewAnalysis} className="new-analysis-btn">
          <RotateCcw size={16} />
          Try Again
        </button>
      </div>
    );
  }

  const confidencePercentage = Math.round(confidence * 100);
  const confidenceColor = confidencePercentage >= 80 ? '#4a7c59' : 
                         confidencePercentage >= 60 ? '#f57c00' : '#d32f2f';
  const confidenceIcon = confidencePercentage >= 80 ? CheckCircle : 
                         confidencePercentage >= 60 ? Target : AlertCircle;
  const ConfidenceIcon = confidenceIcon;

  return (
    <div className="result-container">
      <div className="result-header">
        <Bug className="result-icon" />
        <h3>Identification Result</h3>
      </div>
      
      {pest_name && (
        <>
          <div className="pest-name">
            {pest_name.replace(/_/g, ' ')}
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: '1rem 0' }}>
            <ConfidenceIcon size={16} color={confidenceColor} />
            <span style={{ fontSize: '0.9rem', color: '#6b8e6b' }}>Confidence Level</span>
          </div>
          
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ 
                width: `${confidencePercentage}%`,
                background: `linear-gradient(90deg, ${confidenceColor} 0%, ${confidenceColor}dd 100%)`
              }}
            ></div>
          </div>
          
          <div className="confidence-text">
            {confidencePercentage}% confident
          </div>
          
          {description && (
            <div style={{ 
              marginTop: '1.5rem', 
              padding: '1rem', 
              background: '#f8fdf8', 
              borderRadius: '8px',
              border: '1px solid #e8f5e8'
            }}>
              <h4 style={{ color: '#2d5016', marginBottom: '0.5rem', fontSize: '1rem' }}>
                About this Pest:
              </h4>
              <p style={{ color: '#6b8e6b', fontSize: '0.9rem', lineHeight: '1.5' }}>
                {description}
              </p>
            </div>
          )}
          
          {treatment_suggestions && treatment_suggestions.length > 0 && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              background: '#f0f8f0', 
              borderRadius: '8px',
              border: '1px solid #90c695'
            }}>
              <h4 style={{ color: '#2d5016', marginBottom: '0.75rem', fontSize: '1rem' }}>
                Quick Treatment Tips:
              </h4>
              <ul style={{ color: '#6b8e6b', fontSize: '0.9rem', paddingLeft: '1.2rem' }}>
                {treatment_suggestions.map((suggestion, index) => (
                  <li key={index} style={{ marginBottom: '0.25rem' }}>
                    {suggestion}
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {confidencePercentage < 60 && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              background: '#fff3e0', 
              borderRadius: '8px',
              border: '1px solid #ffcc02'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <AlertCircle size={16} color="#f57c00" />
                <strong style={{ color: '#e65100' }}>Low Confidence Warning</strong>
              </div>
              <p style={{ color: '#ef6c00', fontSize: '0.9rem', margin: 0 }}>
                The identification confidence is below 60%. Consider uploading a clearer image or 
                consulting with our AI assistant for additional guidance.
              </p>
            </div>
          )}
        </>
      )}
      
      <button onClick={onNewAnalysis} className="new-analysis-btn">
        <RotateCcw size={16} />
        Analyze New Image
      </button>
    </div>
  );
};

export default PestResult;