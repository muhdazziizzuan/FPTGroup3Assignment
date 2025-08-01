import React from 'react';
import { Leaf, Bug } from 'lucide-react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo-section">
          <div className="logo">
            <div className="logo-icon">
              <Leaf size={24} className="leaf-icon" />
              <Bug size={16} className="bug-icon" />
            </div>
          </div>
          <div className="logo-text">
            <h1>Organic Farm Pest Management AI</h1>
            <p>Intelligent pest identification & organic treatment recommendations</p>
          </div>
        </div>
        
        <div className="header-stats">
          <div className="stat">
            <span className="stat-number">24/7</span>
            <span className="stat-label">Available</span>
          </div>
          <div className="stat">
            <span className="stat-number">Offline</span>
            <span className="stat-label">Capable</span>
          </div>
          <div className="stat">
            <span className="stat-number">100%</span>
            <span className="stat-label">Organic</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;