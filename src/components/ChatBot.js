import React, { useState, useRef, useEffect, forwardRef, useImperativeHandle } from 'react';
import { Send, Bot, User, Loader } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const ChatBot = forwardRef(({ pestResult }, ref) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: `Hello! I'm your **AgriPest AI Assistant** 🌱

I'm here to help you with:
- **Pest identification** and analysis
- **Organic treatment** recommendations
- **Prevention strategies** for your crops
- **General farming** advice

Feel free to ask me anything about pest management or upload an image for analysis!`,
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentPestContext, setCurrentPestContext] = useState(null);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputMessage]);

  // Update pest context when pestResult changes
  useEffect(() => {
    if (pestResult && pestResult.pest_name && !pestResult.error) {
      setCurrentPestContext(pestResult.pest_name);
    }
  }, [pestResult]);

  const sendMessage = async (message = inputMessage) => {
    if (!message.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Create an AbortController for timeout handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
      
      // Include pest context in the message if available
      let contextualMessage = message;
      if (currentPestContext && !message.toLowerCase().includes(currentPestContext.toLowerCase())) {
        contextualMessage = `Regarding ${currentPestContext}: ${message}`;
      }
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: contextualMessage,
          model: 'phi4-mini'
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: data.response || 'I apologize, but I encountered an issue processing your request. Please try again.',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      let errorMessage = 'I\'m having trouble connecting to my knowledge base. ';
      
      if (error.name === 'AbortError') {
        errorMessage += 'The request timed out. The AI service might be overloaded. Please try again.';
      } else if (error.message.includes('503')) {
        errorMessage += 'The AI service appears to be unavailable. Please make sure Ollama is running and try again.';
      } else if (error.message.includes('404')) {
        errorMessage += 'The AI model is not available. Please check if the required model is installed.';
      } else {
        errorMessage += 'Please check your connection and try again.';
      }
      
      const errorBotMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `⚠️ **Connection Error**\n\n${errorMessage}\n\n*You can still upload images for pest identification while I work on reconnecting.*`,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorBotMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage();
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Expose sendMessage method to parent component
  useImperativeHandle(ref, () => ({
    sendMessage: (message) => {
      // Extract pest name from contextual messages sent by parent
      const pestMatch = message.match(/I've identified a ([^\s]+)/i);
      if (pestMatch) {
        setCurrentPestContext(pestMatch[1]);
      }
      sendMessage(message);
    }
  }));

  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="chatbot-container">
      <div className="chat-messages" ref={messagesContainerRef}>
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.type}`}>
            <div className="message-avatar">
              {message.type === 'user' ? <User size={16} /> : <Bot size={16} />}
            </div>
            <div className="message-content">
              {message.type === 'bot' ? (
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p style={{ margin: '0 0 0.5rem 0', lineHeight: '1.5' }}>{children}</p>,
                    ul: ({ children }) => <ul style={{ margin: '0.5rem 0', paddingLeft: '1.2rem' }}>{children}</ul>,
                    ol: ({ children }) => <ol style={{ margin: '0.5rem 0', paddingLeft: '1.2rem' }}>{children}</ol>,
                    li: ({ children }) => <li style={{ marginBottom: '0.25rem' }}>{children}</li>,
                    strong: ({ children }) => <strong style={{ fontWeight: '600' }}>{children}</strong>,
                    h2: ({ children }) => <h2 style={{ fontSize: '1.1rem', margin: '1rem 0 0.5rem 0', color: message.type === 'user' ? 'inherit' : '#2d5016' }}>{children}</h2>,
                    h3: ({ children }) => <h3 style={{ fontSize: '1rem', margin: '0.75rem 0 0.5rem 0', color: message.type === 'user' ? 'inherit' : '#2d5016' }}>{children}</h3>,
                    code: ({ children }) => (
                      <code style={{ 
                        background: message.type === 'user' ? 'rgba(255,255,255,0.2)' : '#f0f8f0', 
                        padding: '0.2rem 0.4rem', 
                        borderRadius: '4px',
                        fontSize: '0.9rem'
                      }}>
                        {children}
                      </code>
                    )
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              ) : (
                <p>{message.content}</p>
              )}
              <div style={{ 
                fontSize: '0.75rem', 
                opacity: 0.7, 
                marginTop: '0.5rem',
                color: message.type === 'user' ? 'rgba(255,255,255,0.8)' : '#6b8e6b'
              }}>
                {formatTimestamp(message.timestamp)}
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message bot">
            <div className="message-avatar">
              <Loader size={16} className="spinner" style={{ animation: 'spin 1s linear infinite' }} />
            </div>
            <div className="message-content">
              <p style={{ margin: 0, fontStyle: 'italic', color: '#6b8e6b' }}>
                Thinking...
              </p>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <textarea
            ref={textareaRef}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about pest management, treatments, or farming advice..."
            className="chat-input"
            disabled={isLoading}
            rows={1}
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? (
              <Loader size={18} className="spinner" style={{ animation: 'spin 1s linear infinite' }} />
            ) : (
              <Send size={18} />
            )}
          </button>
        </form>
        
        <div style={{ 
          fontSize: '0.75rem', 
          color: '#6b8e6b', 
          textAlign: 'center', 
          marginTop: '0.5rem',
          fontStyle: 'italic'
        }}>
          💡 Tip: Upload an image for pest identification, then ask for treatment advice!
        </div>
      </div>
    </div>
  );
});

ChatBot.displayName = 'ChatBot';

export default ChatBot;