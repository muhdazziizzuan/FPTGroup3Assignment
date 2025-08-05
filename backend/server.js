const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

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

// Pest Classification endpoint
app.post('/api/classify-pest', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    const imagePath = req.file.path;
    const modelPath = path.join(__dirname, '..', 'trainedmodel', 'pest_classifier2.pth');
    const pythonScript = path.join(__dirname, 'pest_classifier.py');

    // Check if model file exists
    if (!fs.existsSync(modelPath)) {
      console.warn('Model file not found, using fallback simulation');
      return simulateClassification(req, res);
    }

    // Check if Python script exists
    if (!fs.existsSync(pythonScript)) {
      console.warn('Python classifier script not found, using fallback simulation');
      return simulateClassification(req, res);
    }

    // Run Python classification script
    const pythonProcess = spawn('python', [pythonScript, imagePath, modelPath]);
    
    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      // Clean up uploaded file
      if (fs.existsSync(imagePath)) {
        fs.unlinkSync(imagePath);
      }

      if (code === 0) {
        try {
          const classificationResult = JSON.parse(result);
          res.json(classificationResult);
        } catch (parseError) {
          console.error('Error parsing Python script output:', parseError);
          res.status(500).json({ 
            error: 'Failed to parse classification result',
            details: parseError.message 
          });
        }
      } else {
        console.error('Python script error:', error);
        // Fallback to simulation if Python script fails
        simulateClassification(req, res, true);
      }
    });

    // Set timeout for Python process
    setTimeout(() => {
      pythonProcess.kill();
      if (fs.existsSync(imagePath)) {
        fs.unlinkSync(imagePath);
      }
      res.status(408).json({ 
        error: 'Classification timeout. Please try again.',
        details: 'The model took too long to process the image' 
      });
    }, 30000); // 30 second timeout

  } catch (error) {
    console.error('Pest classification error:', error);
    
    // Clean up uploaded file on error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({ 
      error: 'Failed to classify pest. Please try again.',
      details: error.message 
    });
  }
});

// Fallback simulation function
function simulateClassification(req, res, isFailover = false) {
  const simulatedResults = [
    { pest_name: 'ants', confidence: 0.92, description: 'Common garden ants that can damage plant roots and farm aphids.' },
    { pest_name: 'bees', confidence: 0.88, description: 'Beneficial pollinators, generally not considered pests.' },
    { pest_name: 'beetle', confidence: 0.85, description: 'Various beetle species that can damage leaves and crops.' },
    { pest_name: 'catterpillar', confidence: 0.90, description: 'Larval stage of moths and butterflies, can cause significant leaf damage.' },
    { pest_name: 'earthworms', confidence: 0.95, description: 'Beneficial soil organisms that improve soil health.' },
    { pest_name: 'earwig', confidence: 0.87, description: 'Nocturnal insects that can damage seedlings and soft plant tissues.' },
    { pest_name: 'grasshopper', confidence: 0.89, description: 'Jumping insects that can cause extensive damage to crops and gardens.' },
    { pest_name: 'moth', confidence: 0.86, description: 'Adult stage of various species, some larvae can be crop pests.' },
    { pest_name: 'slug', confidence: 0.91, description: 'Soft-bodied mollusks that damage leaves and seedlings.' },
    { pest_name: 'snail', confidence: 0.88, description: 'Shelled mollusks that feed on plant material.' },
    { pest_name: 'wasp', confidence: 0.84, description: 'Beneficial predators that control other pest insects.' },
    { pest_name: 'weevil', confidence: 0.93, description: 'Beetle family that includes many serious crop pests.' }
  ];

  const randomResult = simulatedResults[Math.floor(Math.random() * simulatedResults.length)];
  
  const treatmentSuggestions = {
    'ants': ['Use diatomaceous earth around affected areas', 'Apply cinnamon or coffee grounds as natural deterrents', 'Remove food sources and moisture'],
    'beetle': ['Hand-pick beetles in early morning', 'Use row covers during peak season', 'Apply neem oil spray'],
    'catterpillar': ['Inspect plants regularly for eggs', 'Use Bacillus thuringiensis (Bt) spray', 'Encourage beneficial predators'],
    'earwig': ['Remove hiding places like debris', 'Use beer traps', 'Apply diatomaceous earth'],
    'grasshopper': ['Use row covers for protection', 'Apply kaolin clay spray', 'Encourage natural predators'],
    'moth': ['Use pheromone traps', 'Apply beneficial nematodes', 'Remove overwintering sites'],
    'slug': ['Use copper barriers', 'Apply iron phosphate baits', 'Remove hiding places'],
    'snail': ['Hand-pick during evening hours', 'Use beer traps', 'Apply crushed eggshells around plants'],
    'weevil': ['Use beneficial nematodes', 'Apply diatomaceous earth', 'Remove plant debris']
  };

  const result = {
    pest_name: randomResult.pest_name,
    confidence: randomResult.confidence,
    description: randomResult.description,
    treatment_suggestions: treatmentSuggestions[randomResult.pest_name] || ['Consult with agricultural extension services', 'Monitor pest population', 'Consider integrated pest management approaches'],
    simulation_mode: true,
    note: isFailover ? 'Using simulation due to model unavailability' : 'Using simulation mode for demonstration'
  };

  // Clean up uploaded file
  if (req.file && fs.existsSync(req.file.path)) {
    fs.unlinkSync(req.file.path);
  }

  res.json(result);
}

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
      classify_pest: '/api/classify-pest',
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