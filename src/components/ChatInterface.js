import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader, Paperclip, X } from 'lucide-react';
import './ChatInterface.css';

const ChatInterface = ({ pestContext, analysisHistory }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: pestContext 
        ? `Hello! I see you've detected ${pestContext.pestName} with ${(pestContext.confidence * 100).toFixed(1)}% confidence. I'm here to help you with organic treatment options and answer any questions about managing this pest. What would you like to know?`
        : 'Hello! I\'m your organic farm pest management assistant. Upload an image of a pest for AI-powered identification, and I\'ll help you with organic treatments and sustainable farming practices. How can I assist you today?',
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

  // Update welcome message when pest context changes
  useEffect(() => {
    if (pestContext && messages.length === 1) {
      setMessages([{
        id: 1,
        type: 'bot',
        content: `Hello! I see you've detected ${pestContext.pestName} with ${(pestContext.confidence * 100).toFixed(1)}% confidence. I'm here to help you with organic treatment options and answer any questions about managing this pest. What would you like to know?`,
        timestamp: new Date()
      }]);
    }
  }, [pestContext]);

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
    setInputMessage('');
    // Clear attached images state but keep URLs for the message
    setAttachedImages([]);
    setIsLoading(true);
    setShowQuickActions(false);

    // Simulate API call to LLM backend
    try {
      // In real implementation, this would call your backend API
      setTimeout(() => {
        const botResponse = generateMockResponse(inputMessage);
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: botResponse,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
        setIsLoading(false);
      }, 1500);
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
    }
  };

  const getPestTreatment = (pestName) => {
    const treatments = {
      'aphid': 'For aphid control: 1) Spray with neem oil solution (2 tbsp per gallon), 2) Release ladybugs or lacewings, 3) Use insecticidal soap spray, 4) Apply diatomaceous earth around plants, 5) Encourage natural predators with diverse plantings.',
      'caterpillar': 'For caterpillar management: 1) Apply Bacillus thuringiensis (Bt) spray in evening, 2) Hand-pick larger caterpillars, 3) Use row covers during egg-laying season, 4) Encourage birds with nesting boxes, 5) Plant trap crops like nasturtiums.',
      'spider mite': 'For spider mite control: 1) Increase humidity with regular misting, 2) Release predatory mites, 3) Use neem oil or insecticidal soap, 4) Strong water spray to dislodge mites, 5) Improve air circulation around plants.',
      'whitefly': 'For whitefly management: 1) Yellow sticky traps, 2) Neem oil spray application, 3) Encourage beneficial insects like Encarsia wasps, 4) Reflective mulch to confuse adults, 5) Remove heavily infested leaves.',
      'thrips': 'For thrips control: 1) Blue sticky traps (thrips prefer blue), 2) Beneficial predatory mites, 3) Neem oil or spinosad spray, 4) Remove plant debris regularly, 5) Encourage natural predators with flowering plants.'
    };
    return treatments[pestName] || `For ${pestName} control, I recommend consulting with a local agricultural extension office for specific organic treatment options suitable for your region.`;
  };

  const getPestDamage = (pestName) => {
    const damages = {
      'aphid': 'Aphids cause: 1) Yellowing and curling leaves, 2) Stunted plant growth, 3) Sticky honeydew secretion, 4) Sooty mold development, 5) Virus transmission between plants.',
      'caterpillar': 'Caterpillars cause: 1) Holes in leaves and fruits, 2) Complete defoliation in severe cases, 3) Damage to growing tips, 4) Reduced plant vigor, 5) Secondary infections through wounds.',
      'spider mite': 'Spider mites cause: 1) Fine webbing on leaves, 2) Yellow stippling on leaf surface, 3) Bronzing of leaves, 4) Premature leaf drop, 5) Reduced photosynthesis and plant stress.',
      'whitefly': 'Whiteflies cause: 1) Yellowing leaves, 2) Sticky honeydew deposits, 3) Sooty mold growth, 4) Weakened plants, 5) Virus transmission.',
      'thrips': 'Thrips cause: 1) Silver-white streaks on leaves, 2) Black specks (excrement), 3) Distorted growth, 4) Reduced fruit quality, 5) Virus transmission.'
    };
    return damages[pestName] || `${pestName} can cause various types of plant damage. Monitor your plants closely for changes in appearance, growth, or health.`;
  };

  const getPestPrevention = (pestName) => {
    const prevention = {
      'aphid': 'Prevent aphids by: 1) Avoiding over-fertilization with nitrogen, 2) Encouraging beneficial insects with diverse plantings, 3) Regular inspection of plants, 4) Proper plant spacing for air circulation, 5) Companion planting with marigolds or catnip.',
      'caterpillar': 'Prevent caterpillars by: 1) Crop rotation to break lifecycle, 2) Removing plant debris, 3) Encouraging natural predators, 4) Using row covers during vulnerable periods, 5) Regular monitoring for eggs.',
      'spider mite': 'Prevent spider mites by: 1) Maintaining adequate humidity, 2) Avoiding water stress, 3) Regular plant inspection, 4) Proper air circulation, 5) Avoiding dusty conditions.',
      'whitefly': 'Prevent whiteflies by: 1) Quarantining new plants, 2) Regular inspection of plant undersides, 3) Removing weeds that host whiteflies, 4) Proper plant spacing, 5) Using reflective mulches.',
      'thrips': 'Prevent thrips by: 1) Removing plant debris and weeds, 2) Proper sanitation practices, 3) Quarantining new plants, 4) Regular monitoring with blue sticky traps, 5) Maintaining plant health.'
    };
    return prevention[pestName] || `For ${pestName} prevention, focus on maintaining healthy plants, good sanitation, and encouraging beneficial insects in your garden.`;
  };

  const getPestLifecycle = (pestName) => {
    const lifecycles = {
      'aphid': 'Aphid lifecycle: 1) Eggs overwinter on host plants, 2) Wingless females reproduce asexually in spring, 3) Multiple generations per season, 4) Winged forms develop when crowded, 5) Sexual reproduction occurs in fall.',
      'caterpillar': 'Caterpillar lifecycle: 1) Adults lay eggs on host plants, 2) Larvae (caterpillars) feed and grow through 5-6 instars, 3) Pupation occurs in soil or on plants, 4) Adults emerge to mate and lay eggs, 5) 1-3 generations per year depending on species.',
      'spider mite': 'Spider mite lifecycle: 1) Eggs laid on leaf undersides, 2) Six-legged larvae emerge, 3) Two nymphal stages with 8 legs, 4) Adults reproduce rapidly in warm conditions, 5) Complete cycle in 1-2 weeks.',
      'whitefly': 'Whitefly lifecycle: 1) Eggs laid on leaf undersides, 2) Four nymphal stages (scales), 3) Pupation occurs, 4) Adults emerge and fly to new plants, 5) Continuous breeding in warm conditions.',
      'thrips': 'Thrips lifecycle: 1) Eggs inserted into plant tissue, 2) Two larval stages feed on plants, 3) Pupation in soil or plant crevices, 4) Adults emerge and feed/reproduce, 5) Multiple generations per season.'
    };
    return lifecycles[pestName] || `The lifecycle of ${pestName} involves egg, larval/nymphal, and adult stages. Understanding this helps time treatments effectively.`;
  };

  const generateMockResponse = (userInput) => {
    const input = userInput.toLowerCase();
    
    // If we have pest context, provide specific responses
    if (pestContext) {
      const pestName = pestContext.pestName.toLowerCase();
      
      if (input.includes('treatment') || input.includes('control') || input.includes('how to')) {
        return getPestTreatment(pestName);
      } else if (input.includes('damage') || input.includes('symptoms')) {
        return getPestDamage(pestName);
      } else if (input.includes('prevention') || input.includes('prevent')) {
        return getPestPrevention(pestName);
      } else if (input.includes('lifecycle') || input.includes('life cycle')) {
        return getPestLifecycle(pestName);
      }
    }
    
    // General responses based on keywords
    if (input.includes('aphid')) {
      return 'Aphids are common soft-bodied insects that feed on plant sap. For organic treatment, I recommend: 1) Neem oil spray (mix 2 tbsp per gallon of water), 2) Introduce beneficial insects like ladybugs, 3) Use insecticidal soap, 4) Plant companion crops like marigolds or catnip. Apply treatments in early morning or evening to avoid harming beneficial insects.';
    } else if (input.includes('caterpillar') || input.includes('worm')) {
      return 'For caterpillars, try these organic solutions: 1) Bacillus thuringiensis (Bt) spray - safe for beneficial insects, 2) Hand-picking for small infestations, 3) Row covers during egg-laying periods, 4) Encourage birds and beneficial wasps. Apply Bt in the evening when caterpillars are most active.';
    } else if (input.includes('spider mite')) {
      return 'Spider mites thrive in hot, dry conditions. Organic treatments include: 1) Increase humidity around plants, 2) Predatory mites like Phytoseiulus persimilis, 3) Neem oil or insecticidal soap spray, 4) Strong water spray to dislodge mites. Ensure good air circulation and avoid over-fertilizing with nitrogen.';
    } else if (input.includes('prevention') || input.includes('prevent')) {
      return 'Prevention is key in organic farming! Here are essential practices: 1) Crop rotation to break pest cycles, 2) Companion planting (basil with tomatoes, marigolds throughout), 3) Maintain soil health with compost, 4) Encourage beneficial insects with diverse plantings, 5) Regular monitoring and early intervention, 6) Proper spacing for air circulation.';
    } else {
      const contextMessage = pestContext 
        ? `Based on the detected ${pestContext.pestName}, I can help with specific treatment options. `
        : '';
      return contextMessage + 'I understand you\'re asking about pest management. Could you provide more specific details about the pest you\'re dealing with or the symptoms you\'re observing? For example, describe the insect\'s appearance, which plants are affected, or the type of damage you\'re seeing. This will help me provide more targeted organic treatment recommendations.';
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