import React, { useState, useRef } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import ChatBot from './components/ChatBot';
import PestResult from './components/PestResult';
import ModelSelector from './components/ModelSelector';
import { Leaf, Bug, MessageCircle, Camera } from 'lucide-react';

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [pestResult, setPestResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState('pest_classifier2.pth');
  const chatBotRef = useRef(null);

  const handleImageUpload = async (file) => {
    setUploadedImage(file);
    setIsAnalyzing(true);
    setPestResult(null);

    try {
      // Create FormData for image upload
      const formData = new FormData();
      formData.append('image', file);
      formData.append('model', selectedModel);

      // Send image to backend for pest classification
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
      
      const response = await fetch('/api/classify-pest', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error('Failed to classify pest');
      }

      const result = await response.json();
      setPestResult(result);

      // Send result to chatbot for contextual advice
      if (chatBotRef.current && result.pest_name) {
        const contextMessage = `I've identified a ${result.pest_name} in the uploaded image with ${(result.confidence * 100).toFixed(1)}% confidence. Can you provide treatment recommendations and prevention tips for this pest?`;
        chatBotRef.current.sendMessage(contextMessage);
      }
    } catch (error) {
      console.error('Error classifying pest:', error);
      
      let errorMessage = 'Failed to classify pest. Please try again.';
      if (error.name === 'AbortError') {
        errorMessage = 'Image analysis timed out. The service might be overloaded. Please try again.';
      }
      
      setPestResult({
        error: errorMessage,
        pest_name: null,
        confidence: 0
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleNewAnalysis = () => {
    setUploadedImage(null);
    setPestResult(null);
    setIsAnalyzing(false);
  };

  const handleModelChange = (newModel) => {
    setSelectedModel(newModel);
    // Clear uploaded image and results when model changes
    setUploadedImage(null);
    setPestResult(null);
    setIsAnalyzing(false);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <Bug className="logo-icon" />
            <h1>AgriPest AI</h1>
            <Leaf className="leaf-icon" />
          </div>
          <p className="tagline">Intelligent Pest Identification & Management</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="content-grid">
          {/* Left Panel - Image Upload & Results */}
          <div className="left-panel">
            <div className="panel-header">
              <Camera className="panel-icon" />
              <h2>Pest Analysis</h2>
            </div>
            
            <ModelSelector 
              selectedModel={selectedModel}
              onModelChange={handleModelChange}
              isAnalyzing={isAnalyzing}
            />
            
            <ImageUpload 
              onImageUpload={handleImageUpload}
              uploadedImage={uploadedImage}
              isAnalyzing={isAnalyzing}
            />
            
            {pestResult && (
              <PestResult 
                result={pestResult}
                onNewAnalysis={handleNewAnalysis}
              />
            )}
          </div>

          {/* Right Panel - ChatBot */}
          <div className="right-panel">
            <div className="panel-header">
              <MessageCircle className="panel-icon" />
              <h2>AI Assistant</h2>
            </div>
            
            <ChatBot ref={chatBotRef} pestResult={pestResult} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>© 2024 AgriPest AI - Sustainable Farming Solutions</p>
      </footer>
    </div>
  );
}

export default App;