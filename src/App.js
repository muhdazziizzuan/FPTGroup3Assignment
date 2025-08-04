import React, { useState } from 'react';
import './App.css';
import ChatInterface from './components/ChatInterface';
import ImageUpload from './components/ImageUpload';
import PestResults from './components/PestResults';

function App() {
  const [pestResults, setPestResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState([]);

  // Mock CNN model analysis function
  const analyzePestImage = async (imageFile) => {
    setIsAnalyzing(true);
    
    try {
      // Simulate API call to your trained CNN model
      // Replace this with actual API call to your backend
      await new Promise(resolve => setTimeout(resolve, 3000)); // Simulate processing time
      
      // Mock response - replace with actual CNN model results
      const mockResults = {
        pestName: "Aphids (Green Peach Aphid)",
        confidence: 0.92,
        severity: "Medium",
        description: "Small, soft-bodied insects that feed on plant sap. They cluster on new growth, undersides of leaves, and stems. Can transmit plant viruses and cause yellowing, curling, and stunted growth.",
        actionRequired: "Monitor closely and begin organic treatment within 24-48 hours to prevent population explosion.",
        organicTreatments: [
          "Spray with insecticidal soap solution (2-3% concentration) every 3-4 days",
          "Release beneficial insects like ladybugs or lacewings",
          "Apply neem oil spray in early morning or evening",
          "Use reflective mulch to confuse and deter aphids",
          "Encourage natural predators with companion planting (marigolds, catnip)"
        ],
        prevention: [
          "Regular inspection of plants, especially new growth",
          "Avoid over-fertilizing with nitrogen",
          "Maintain proper plant spacing for air circulation",
          "Remove weeds that can harbor aphids"
        ],
        timeline: "2-3 weeks with consistent organic treatment",
        riskFactors: ["High humidity", "Overcrowded plants", "Stressed plants"]
      };
      
      setPestResults(mockResults);
      setAnalysisHistory(prev => [...prev, {
        timestamp: new Date(),
        results: mockResults,
        imageFile: imageFile
      }]);
      
    } catch (error) {
      console.error('Error analyzing image:', error);
      // Handle error appropriately
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="App">
      <main className="main-content">
        <div className="container">
          <div className="side-by-side-layout">
            <div className="left-section">
              <ImageUpload 
                onImageAnalysis={analyzePestImage}
                isLoading={isAnalyzing}
              />
              
              {pestResults && (
                <PestResults results={pestResults} />
              )}
            </div>
            
            <div className="right-section">
              <ChatInterface 
                pestContext={pestResults}
                analysisHistory={analysisHistory}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;