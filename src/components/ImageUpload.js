import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Camera, X, Loader, AlertCircle } from 'lucide-react';
import './ImageUpload.css';

const ImageUpload = ({ onImageAnalysis, isLoading }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setError(null);
    
    if (rejectedFiles.length > 0) {
      setError('Please upload a valid image file (JPG, PNG, or WebP)');
      return;
    }

    const file = acceptedFiles[0];
    if (file) {
      // Check file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }

      setSelectedImage(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    multiple: false,
    disabled: isLoading
  });

  const handleAnalyze = () => {
    if (selectedImage && onImageAnalysis) {
      onImageAnalysis(selectedImage);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setError(null);
  };

  const handleCameraCapture = (e) => {
    const file = e.target.files[0];
    if (file) {
      onDrop([file], []);
    }
  };

  return (
    <div className="image-upload">
      <div className="upload-header">
        <h2>Pest Image Analysis</h2>
        <p>Upload or capture an image of the pest for AI-powered identification and organic treatment recommendations</p>
      </div>

      {!imagePreview ? (
        <div className="upload-section">
          <div 
            {...getRootProps()} 
            className={`dropzone ${isDragActive ? 'active' : ''} ${isLoading ? 'disabled' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="dropzone-content">
              <Upload className="upload-icon" size={48} />
              <h3>Drop your pest image here</h3>
              <p>or click to browse files</p>
              <div className="file-info">
                <span>Supports: JPG, PNG, WebP</span>
                <span>Max size: 10MB</span>
              </div>
            </div>
          </div>

          <div className="upload-divider">
            <span>or</span>
          </div>

          <div className="camera-section">
            <label htmlFor="camera-input" className="camera-button">
              <Camera size={20} />
              Take Photo
            </label>
            <input
              id="camera-input"
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleCameraCapture}
              style={{ display: 'none' }}
              disabled={isLoading}
            />
          </div>

          {error && (
            <div className="error-message">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
        </div>
      ) : (
        <div className="preview-section">
          <div className="image-preview">
            <img src={imagePreview} alt="Pest preview" />
            <button 
              className="remove-button" 
              onClick={handleRemoveImage}
              disabled={isLoading}
            >
              <X size={16} />
            </button>
          </div>

          <div className="image-info">
            <h3>{selectedImage.name}</h3>
            <p>Size: {(selectedImage.size / 1024 / 1024).toFixed(2)} MB</p>
            <p>Type: {selectedImage.type}</p>
          </div>

          <div className="analysis-section">
            <button 
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader className="spinner" size={20} />
                  Analyzing Image...
                </>
              ) : (
                'Analyze Pest'
              )}
            </button>
            
            {isLoading && (
              <div className="analysis-progress">
                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>
                <p>AI is processing your image and identifying the pest...</p>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="upload-tips">
        <h4>Tips for better results:</h4>
        <ul>
          <li>Take clear, well-lit photos</li>
          <li>Focus on the pest, not just the damage</li>
          <li>Include some surrounding plant context</li>
          <li>Avoid blurry or distant shots</li>
          <li>Multiple angles can help with identification</li>
        </ul>
      </div>
    </div>
  );
};

export default ImageUpload;