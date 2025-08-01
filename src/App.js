import React from 'react';
import './App.css';
import Header from './components/Header';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="App">
      <Header />
      
      <main className="main-content">
        <div className="container">
          <ChatInterface />
        </div>
      </main>
    </div>
  );
}

export default App;