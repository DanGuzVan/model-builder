import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

function ResultsPanel({ trainingResults, optimizationResults }) {
  const [activeView, setActiveView] = useState('training');

  const renderTrainingResults = () => {
    if (!trainingResults || !trainingResults.metrics) {
      return (
        <div className="empty-state">
          <div className="empty-state-icon">📊</div>
          <p>No training results to display</p>
          <p style={{ fontSize: '0.875rem', color: '#888' }}>
            Complete a training run to see results here
          </p>
        </div>
      );
    }

    const metrics = trainingResults.metrics;
    const chartData = metrics.train_loss.map((_, index) => ({
      epoch: index + 1,
      trainLoss: metrics.train_loss[index],
      valLoss: metrics.val_loss[index],
      trainAcc: metrics.train_accuracy[index] * 100,
      valAcc: metrics.val_accuracy[index] * 100,
    }));

    return (
      <div>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">
              {(trainingResults.best_accuracy * 100).toFixed(2)}%
            </div>
            <div className="metric-label">Best Accuracy</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">
              {metrics.train_loss[metrics.train_loss.length - 1].toFixed(4)}
            </div>
            <div className="metric-label">Final Train Loss</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">
              {metrics.val_loss[metrics.val_loss.length - 1].toFixed(4)}
            </div>
            <div className="metric-label">Final Val Loss</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{trainingResults.epochs}</div>
            <div className="metric-label">Total Epochs</div>
          </div>
        </div>

        <h4 style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>Loss Curves</h4>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'bottom' }} />
              <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="trainLoss"
                stroke="#667eea"
                name="Train Loss"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="valLoss"
                stroke="#f59e0b"
                name="Val Loss"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <h4 style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>Accuracy Curves</h4>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'bottom' }} />
              <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="trainAcc"
                stroke="#22c55e"
                name="Train Accuracy"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="valAcc"
                stroke="#ef4444"
                name="Val Accuracy"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  const renderOptimizationResults = () => {
    if (!optimizationResults) {
      return (
        <div className="empty-state">
          <div className="empty-state-icon">🔬</div>
          <p>No optimization results to display</p>
          <p style={{ fontSize: '0.875rem', color: '#888' }}>
            Complete an optimization run to see results here
          </p>
        </div>
      );
    }

    const history = optimizationResults.history || [];
    const chartData = history.map((entry) => ({
      iteration: entry.iteration,
      bestFitness: entry.best_fitness * 100,
      avgFitness: entry.avg_fitness * 100,
    }));

    const bestResult = optimizationResults.best_result;

    return (
      <div>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">
              {bestResult ? `${(bestResult.accuracy * 100).toFixed(2)}%` : '-'}
            </div>
            <div className="metric-label">Best Accuracy</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">
              {optimizationResults.algorithm.toUpperCase()}
            </div>
            <div className="metric-label">Algorithm</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{history.length}</div>
            <div className="metric-label">Iterations</div>
          </div>
        </div>

        {bestResult && (
          <div className="panel" style={{ marginTop: '1.5rem', background: '#f9f9f9' }}>
            <h4 style={{ marginBottom: '1rem' }}>Best Network Configuration</h4>
            <div className="grid grid-4">
              <div>
                <strong>Input Size:</strong>
                <br />
                {bestResult.network_config.input_size}
              </div>
              <div>
                <strong>Hidden Layers:</strong>
                <br />
                [{bestResult.network_config.hidden_layers.join(', ')}]
              </div>
              <div>
                <strong>Output Size:</strong>
                <br />
                {bestResult.network_config.output_size}
              </div>
              <div>
                <strong>Dropout:</strong>
                <br />
                {bestResult.network_config.dropout.toFixed(2)}
              </div>
            </div>
          </div>
        )}

        {chartData.length > 0 && (
          <>
            <h4 style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>Fitness Evolution</h4>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'bottom' }} />
                  <YAxis label={{ value: 'Fitness (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="bestFitness"
                    stroke="#667eea"
                    name="Best Fitness"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="avgFitness"
                    stroke="#f59e0b"
                    name="Avg Fitness"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </div>
    );
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Results</h2>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className={`btn ${activeView === 'training' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveView('training')}
          >
            Training
          </button>
          <button
            className={`btn ${activeView === 'optimization' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveView('optimization')}
          >
            Optimization
          </button>
        </div>
      </div>

      {activeView === 'training' ? renderTrainingResults() : renderOptimizationResults()}
    </div>
  );
}

export default ResultsPanel;
