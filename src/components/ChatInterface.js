import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader, Paperclip, X } from 'lucide-react';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m your organic farm pest management assistant. I can help you identify pests, recommend organic treatments, and answer questions about sustainable farming practices. How can I assist you today?',
      timestamp: new Date()
    }
  ]);
  const [showQuickActions, setShowQuickActions] = useState(true);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [attachedImages, setAttachedImages] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Cleanup object URLs when component unmounts
  useEffect(() => {
    return () => {
      attachedImages.forEach(img => URL.revokeObjectURL(img.url));
    };
  }, []);

  const handleImageAttach = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    handleFiles(files);
  };

  const handleFiles = (files) => {
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    const newImages = imageFiles.map(file => ({
      id: Date.now() + Math.random(),
      file,
      url: URL.createObjectURL(file),
      name: file.name
    }));
    setAttachedImages(prev => [...prev, ...newImages]);
  };

  const removeImage = (imageId) => {
    setAttachedImages(prev => {
      const updated = prev.filter(img => img.id !== imageId);
      // Clean up object URLs
      const removed = prev.find(img => img.id === imageId);
      if (removed) {
        URL.revokeObjectURL(removed.url);
      }
      return updated;
    });
  };

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
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const quickActions = [
    { id: 1, text: "🐛 Identify this pest", icon: "🔍" },
    { id: 2, text: "🌱 Organic treatment options", icon: "💚" },
    { id: 3, text: "🚨 Emergency pest problem", icon: "⚡" },
    { id: 4, text: "📅 Seasonal pest calendar", icon: "📊" },
    { id: 5, text: "🌿 Prevention strategies", icon: "🛡️" },
    { id: 6, text: "📍 Local pest alerts", icon: "🗺️" }
  ];

  const handleQuickAction = (action) => {
    setInputMessage(action.text);
    setShowQuickActions(false);
    inputRef.current?.focus();
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if ((!inputMessage.trim() && attachedImages.length === 0) || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      images: attachedImages.map(img => ({ url: img.url, name: img.name })),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentMessage = inputMessage;
    const currentImages = [...attachedImages];
    setInputMessage('');
    setAttachedImages([]);
    setIsLoading(true);
    setShowQuickActions(false);

    try {
      // Prepare form data for API call
      const formData = new FormData();
      formData.append('message', currentMessage);
      
      // Add images to form data
      currentImages.forEach((img, index) => {
        formData.append('images', img.file);
      });

      // Call backend API
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: data.response,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
      
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Show error message to user
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error while processing your request. Please make sure the backend service is running and try again.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div 
      className={`chat-interface ${isDragOver ? 'drag-over' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="chat-header">
        <div>
          <h3>AI Pest Management Assistant</h3>
          <p>Ask me anything about organic pest control</p>
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.type}`}>
            <div className="message-avatar">
              {message.type === 'bot' ? <Bot size={20} /> : <User size={20} />}
            </div>
            <div className="message-content">
              {message.images && message.images.length > 0 && (
                <div className="message-images">
                  {message.images.map((img, index) => (
                    <img 
                      key={index} 
                      src={img.url} 
                      alt={img.name}
                      className="message-image"
                    />
                  ))}
                </div>
              )}
              <div className="message-text">{message.content}</div>
              <div className="message-time">{formatTime(message.timestamp)}</div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message bot">
            <div className="message-avatar">
              <Bot size={20} />
            </div>
            <div className="message-content">
              <div className="message-text loading">
                <Loader className="spinner" size={16} />
                Thinking...
              </div>
            </div>
          </div>
        )}
        
        {showQuickActions && messages.length === 1 && (
          <div className="quick-actions">
            <div className="quick-actions-header">
              <h4>Quick Actions</h4>
              <p>Get started with these common requests:</p>
            </div>
            <div className="quick-actions-grid">
              {quickActions.map((action) => (
                <button
                  key={action.id}
                  className="quick-action-btn"
                  onClick={() => handleQuickAction(action)}
                >
                  <span className="quick-action-icon">{action.icon}</span>
                  <span className="quick-action-text">{action.text}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSendMessage}>
        {attachedImages.length > 0 && (
          <div className="attached-images">
            {attachedImages.map((image) => (
              <div key={image.id} className="attached-image">
                <img src={image.url} alt={image.name} />
                <span className="image-name">{image.name}</span>
                <button
                  type="button"
                  className="remove-image-btn"
                  onClick={() => removeImage(image.id)}
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
        
        <div className="input-wrapper">
          <div className="chat-input-container">
            <button
              type="button"
              className="attach-button"
              onClick={handleImageAttach}
              title="Attach images"
            >
              <Paperclip size={20} />
            </button>
            
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask about pest identification, organic treatments, or farming advice..."
              className="chat-input"
              disabled={isLoading}
            />
            
            <button
              type="submit"
              className="send-button"
              disabled={(!inputMessage.trim() && attachedImages.length === 0) || isLoading}
              title="Send message"
            >
              {isLoading ? <Loader className="spinner" size={20} /> : <Send size={20} />}
            </button>
          </div>
          
          {!inputMessage && attachedImages.length === 0 && (
            <div className="input-suggestions">
              <span className="suggestion-text">💡 Try: "What pest is this?", "Organic aphid treatment", or attach a photo</span>
            </div>
          )}
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
      </form>
    </div>
  );
};

export default ChatInterface;