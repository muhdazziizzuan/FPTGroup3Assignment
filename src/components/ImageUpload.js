import React, { useState, useRef } from 'react';
import { Upload, Image, RotateCcw } from 'lucide-react';

const ImageUpload = ({ onImageUpload, uploadedImage, isAnalyzing }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileSelect = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (JPG, PNG, GIF, etc.)');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    onImageUpload(file);
  };

  const handleUploadAreaClick = () => {
    if (!isAnalyzing) {
      fileInputRef.current?.click();
    }
  };

  const handleNewUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onImageUpload(null);
  };

  return (
    <div className="upload-container">
      {!uploadedImage ? (
        <div
          className={`upload-area ${isDragOver ? 'drag-over' : ''} ${isAnalyzing ? 'loading' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleUploadAreaClick}
        >
          <Upload className="upload-icon" />
          <div className="upload-text">
            Drop your pest image here
          </div>
          <div className="upload-subtext">
            or click to browse files
          </div>
          <div className="upload-subtext" style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
            Supports JPG, PNG, GIF • Max 10MB
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInputChange}
            className="file-input"
            disabled={isAnalyzing}
          />
        </div>
      ) : (
        <div className="image-preview">
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <img
              src={URL.createObjectURL(uploadedImage)}
              alt="Uploaded pest"
              className="preview-image"
            />
            
            {isAnalyzing && (
              <div className="analyzing-overlay">
                <div className="spinner"></div>
                <div style={{ color: '#4a7c59', fontWeight: '600' }}>
                  Analyzing pest...
                </div>
                <div style={{ color: '#6b8e6b', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  This may take a few moments
                </div>
              </div>
            )}
          </div>
          
          {!isAnalyzing && (
            <button
              onClick={handleNewUpload}
              className="new-analysis-btn"
              style={{ marginTop: '1rem' }}
            >
              <RotateCcw size={16} />
              Analyze New Image
            </button>
          )}
        </div>
      )}
      
      {uploadedImage && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#f8fdf8', borderRadius: '8px', border: '1px solid #e8f5e8' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <Image size={16} color="#4a7c59" />
            <strong style={{ color: '#2d5016' }}>Image Details:</strong>
          </div>
          <div style={{ fontSize: '0.9rem', color: '#6b8e6b' }}>
            <div><strong>Name:</strong> {uploadedImage.name}</div>
            <div><strong>Size:</strong> {(uploadedImage.size / 1024 / 1024).toFixed(2)} MB</div>
            <div><strong>Type:</strong> {uploadedImage.type}</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;