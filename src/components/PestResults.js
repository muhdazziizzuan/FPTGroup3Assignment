import React from 'react';
import { Bug, AlertTriangle, CheckCircle, Leaf, Clock, Target } from 'lucide-react';
import './PestResults.css';

const PestResults = ({ results }) => {
  if (!results) return null;

  const getSeverityIcon = (severity) => {
    switch (severity.toLowerCase()) {
      case 'high':
        return <AlertTriangle className="severity-icon high" />;
      case 'medium':
        return <AlertTriangle className="severity-icon medium" />;
      case 'low':
        return <CheckCircle className="severity-icon low" />;
      default:
        return <AlertTriangle className="severity-icon medium" />;
    }
  };

  const getSeverityClass = (severity) => {
    return `severity-${severity.toLowerCase()}`;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return '#22c55e';
    if (confidence >= 0.7) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="pest-results">
      <div className="results-header">
        <div className="pest-identification">
          <div className="pest-icon">
            <Bug size={32} />
          </div>
          <div className="pest-info">
            <h2>{results.pestName}</h2>
            <div className="confidence-score">
              <span>Confidence: </span>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill"
                  style={{ 
                    width: `${results.confidence * 100}%`,
                    backgroundColor: getConfidenceColor(results.confidence)
                  }}
                ></div>
              </div>
              <span className="confidence-text">
                {(results.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        <div className={`severity-badge ${getSeverityClass(results.severity)}`}>
          {getSeverityIcon(results.severity)}
          <span>{results.severity} Risk</span>
        </div>
      </div>

      <div className="results-content">
        <div className="description-card">
          <h3>
            <Target className="card-icon" />
            Pest Description
          </h3>
          <p>{results.description}</p>
        </div>

        <div className="action-card">
          <h3>
            <Clock className="card-icon" />
            Immediate Action Required
          </h3>
          <div className="action-alert">
            <AlertTriangle size={20} />
            <span>{results.actionRequired}</span>
          </div>
        </div>

        <div className="treatments-card">
          <h3>
            <Leaf className="card-icon" />
            Organic Treatment Options
          </h3>
          <div className="treatments-list">
            {results.organicTreatments.map((treatment, index) => (
              <div key={index} className="treatment-item">
                <div className="treatment-number">{index + 1}</div>
                <div className="treatment-text">{treatment}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="additional-info">
          <div className="info-grid">
            <div className="info-item">
              <h4>Treatment Priority</h4>
              <p className={getSeverityClass(results.severity)}>
                {results.severity} Priority
              </p>
            </div>
            <div className="info-item">
              <h4>Identification Accuracy</h4>
              <p style={{ color: getConfidenceColor(results.confidence) }}>
                {results.confidence >= 0.9 ? 'Very High' : 
                 results.confidence >= 0.7 ? 'High' : 'Moderate'}
              </p>
            </div>
            <div className="info-item">
              <h4>Organic Approach</h4>
              <p className="organic-badge">
                <Leaf size={16} />
                100% Organic Solutions
              </p>
            </div>
          </div>
        </div>

        <div className="next-steps">
          <h3>Recommended Next Steps</h3>
          <div className="steps-list">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h4>Immediate Treatment</h4>
                <p>Apply the first recommended organic treatment as soon as possible.</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h4>Monitor Progress</h4>
                <p>Check the affected plants daily for 3-5 days to assess treatment effectiveness.</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h4>Prevention</h4>
                <p>Implement preventive measures to avoid future infestations.</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">4</div>
              <div className="step-content">
                <h4>Follow-up</h4>
                <p>If symptoms persist after 1 week, consider alternative treatments or consult local experts.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="disclaimer">
          <AlertTriangle size={16} />
          <p>
            <strong>Disclaimer:</strong> This AI identification is for guidance only. 
            For severe infestations or if you're unsure about the identification, 
            please consult with local agricultural extension services or organic farming experts.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PestResults;