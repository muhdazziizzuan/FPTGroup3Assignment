const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8000;
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://ollama:11434';

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Configure multer for file uploads
const upload = multer({
  dest: 'uploads/',
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'), false);
    }
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Check Ollama connection
app.get('/ollama/status', async (req, res) => {
  try {
    const response = await axios.get(`${OLLAMA_URL}/api/tags`);
    res.json({ status: 'connected', models: response.data.models });
  } catch (error) {
    res.status(500).json({ 
      status: 'disconnected', 
      error: error.message 
    });
  }
});

// Chat endpoint
// Chat endpoint
app.post('/api/chat', upload.array('images', 5), async (req, res) => {
  try {
    const { message, model = 'llama3.2:1b' } = req.body;
    const images = req.files || [];

    if (!message && images.length === 0) {
      return res.status(400).json({ error: 'Message or images required' });
    }

    // Prepare the prompt with context about pest management
    const systemPrompt = `You are an expert organic farm pest management assistant. You help farmers identify pests, recommend organic treatments, and provide sustainable farming advice. 

    IMPORTANT: Format your responses using markdown for better readability:
    - Use **bold text** for important points and pest names
    - Use bullet points (- or *) for lists of treatments or symptoms
    - Use numbered lists (1. 2. 3.) for step-by-step instructions
    - Use ## headings for main sections like "Identification" or "Treatment"
    - Keep paragraphs short and well-spaced

    Always provide practical, safe, and environmentally friendly solutions. If images are provided, analyze them for pest identification.`;

    let fullPrompt = `${systemPrompt}\n\nUser: ${message}`;

    // If images are provided, add image analysis context
    if (images.length > 0) {
      fullPrompt += `\n\nThe user has provided ${images.length} image(s) for analysis. Please analyze the image(s) for pest identification and provide relevant organic treatment recommendations.`;
    }

    // Prepare request to Ollama
    const ollamaRequest = {
      model: model,
      prompt: fullPrompt,
      stream: false,
      options: {
        temperature: 0.7,
        top_p: 0.9,
        num_predict: 1000
      }
    };

    // If images are provided, convert them to base64 and include in request
    if (images.length > 0) {
      ollamaRequest.images = [];
      for (const image of images) {
        const imageBuffer = fs.readFileSync(image.path);
        const base64Image = imageBuffer.toString('base64');
        ollamaRequest.images.push(base64Image);
        
        // Clean up uploaded file
        fs.unlinkSync(image.path);
      }
    }

    // Call Ollama API
    const response = await axios.post(`${OLLAMA_URL}/api/generate`, ollamaRequest, {
      timeout: 300000, // 5 minute timeout for model generation
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Clean up any remaining uploaded files
    images.forEach(image => {
      if (fs.existsSync(image.path)) {
        fs.unlinkSync(image.path);
      }
    });

    res.json({
      response: response.data.response,
      model: model,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Chat API error:', error);
    
    // Clean up uploaded files on error
    if (req.files) {
      req.files.forEach(file => {
        if (fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
      });
    }

    if (error.code === 'ECONNREFUSED') {
      res.status(503).json({ 
        error: 'Ollama service is not available. Please check if Ollama is running.' 
      });
    } else if (error.response?.status === 404) {
      res.status(404).json({ 
        error: `Model not found. Please ensure the model is pulled in Ollama.` 
      });
    } else {
      res.status(500).json({ 
        error: 'Internal server error',
        details: error.message 
      });
    }
  }
});

// List available models
app.get('/api/models', async (req, res) => {
  try {
    const response = await axios.get(`${OLLAMA_URL}/api/tags`);
    res.json(response.data);
  } catch (error) {
    console.error('Models API error:', error);
    res.status(500).json({ 
      error: 'Failed to fetch models',
      details: error.message 
    });
  }
});

// Pull a model
app.post('/api/models/pull', async (req, res) => {
  try {
    const { model } = req.body;
    
    if (!model) {
      return res.status(400).json({ error: 'Model name is required' });
    }

    const response = await axios.post(`${OLLAMA_URL}/api/pull`, { name: model }, {
      timeout: 300000 // 5 minute timeout for model pulling
    });

    res.json({ message: `Model ${model} pulled successfully` });
  } catch (error) {
    console.error('Pull model error:', error);
    res.status(500).json({ 
      error: 'Failed to pull model',
      details: error.message 
    });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({ 
    message: 'Organic Farm Pest Management API',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      chat: '/api/chat',
      models: '/api/models',
      ollama_status: '/ollama/status'
    }
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large. Maximum size is 10MB.' });
    }
    if (error.code === 'LIMIT_FILE_COUNT') {
      return res.status(400).json({ error: 'Too many files. Maximum is 5 images.' });
    }
  }
  
  console.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Backend server running on port ${PORT}`);
  console.log(`Ollama URL: ${OLLAMA_URL}`);
});