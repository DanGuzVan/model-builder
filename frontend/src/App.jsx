import { useState, useEffect } from 'react';
import DatasetPanel from './components/DatasetPanel';
import NetworkBuilder from './components/NetworkBuilder';
import TrainingPanel from './components/TrainingPanel';
import OptimizationPanel from './components/OptimizationPanel';
import ResultsPanel from './components/ResultsPanel';
import { healthApi } from './api';

function App() {
  const [activeTab, setActiveTab] = useState('datasets');
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedNetwork, setSelectedNetwork] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null);
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthApi.check();
        setApiStatus('connected');
      } catch (error) {
        setApiStatus('disconnected');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'datasets', label: 'Datasets' },
    { id: 'networks', label: 'Network Builder' },
    { id: 'training', label: 'Training' },
    { id: 'optimization', label: 'Optimization' },
    { id: 'results', label: 'Results' },
  ];

  return (
    <div className="app">
      <header className="app-header">
        <h1>NN Optimizer</h1>
        <div className="status-indicator">
          <span className={`status-dot ${apiStatus}`}></span>
          <span>API: {apiStatus}</span>
        </div>
      </header>

      <nav className="app-nav">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {activeTab === 'datasets' && (
          <DatasetPanel
            selectedDataset={selectedDataset}
            onSelectDataset={setSelectedDataset}
          />
        )}

        {activeTab === 'networks' && (
          <NetworkBuilder
            selectedDataset={selectedDataset}
            selectedNetwork={selectedNetwork}
            onSelectNetwork={setSelectedNetwork}
          />
        )}

        {activeTab === 'training' && (
          <TrainingPanel
            selectedDataset={selectedDataset}
            selectedNetwork={selectedNetwork}
            onTrainingComplete={setTrainingResults}
          />
        )}

        {activeTab === 'optimization' && (
          <OptimizationPanel
            selectedDataset={selectedDataset}
            onOptimizationComplete={setOptimizationResults}
          />
        )}

        {activeTab === 'results' && (
          <ResultsPanel
            trainingResults={trainingResults}
            optimizationResults={optimizationResults}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>Neural Network Optimizer - PSO & Genetic Algorithm Optimization</p>
      </footer>
    </div>
  );
}

export default App;
