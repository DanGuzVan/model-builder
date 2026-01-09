import { useState, useEffect } from 'react';
import { trainingApi } from '../api';

function TrainingPanel({ selectedDataset, selectedNetwork, onTrainingComplete }) {
  const [trainingRuns, setTrainingRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const [formData, setFormData] = useState({
    learningRate: 0.001,
    batchSize: 32,
    epochs: 100,
  });

  useEffect(() => {
    loadTrainingRuns();
    const interval = setInterval(loadTrainingRuns, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadTrainingRuns = async () => {
    try {
      setLoading(true);
      const response = await trainingApi.list();
      setTrainingRuns(response.training_runs || []);
      setError(null);
    } catch (err) {
      setError('Failed to load training runs');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedDataset || !selectedNetwork) {
      setError('Please select a dataset and network first');
      return;
    }

    try {
      setSubmitting(true);
      setError(null);

      const runData = {
        dataset_id: selectedDataset.id,
        network_id: selectedNetwork.id,
        learning_rate: formData.learningRate,
        batch_size: formData.batchSize,
        epochs: formData.epochs,
      };

      await trainingApi.create(runData);
      await loadTrainingRuns();
    } catch (err) {
      setError('Failed to start training');
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = async (id) => {
    try {
      await trainingApi.cancel(id);
      await loadTrainingRuns();
    } catch (err) {
      setError('Failed to cancel training');
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this training run?')) return;

    try {
      await trainingApi.delete(id);
      await loadTrainingRuns();
    } catch (err) {
      setError('Failed to delete training run');
    }
  };

  const handleViewResults = (run) => {
    onTrainingComplete(run);
  };

  const getStatusBadge = (status) => {
    const statusClasses = {
      pending: 'badge-pending',
      running: 'badge-running',
      completed: 'badge-completed',
      failed: 'badge-failed',
      cancelled: 'badge-cancelled',
    };
    return <span className={`badge ${statusClasses[status] || ''}`}>{status}</span>;
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Training</h2>
        <button className="btn btn-secondary" onClick={loadTrainingRuns}>
          Refresh
        </button>
      </div>

      {error && <div className="message message-error">{error}</div>}

      {(!selectedDataset || !selectedNetwork) && (
        <div className="message message-info">
          Please select a dataset and network to start training.
        </div>
      )}

      {/* Training Form */}
      <form onSubmit={handleSubmit} className="panel" style={{ background: '#f9f9f9' }}>
        <h3 style={{ marginBottom: '1rem' }}>Start New Training</h3>

        <div className="grid grid-3">
          <div className="form-group">
            <label className="form-label">Learning Rate</label>
            <input
              type="number"
              className="form-input"
              min="0.0001"
              max="1"
              step="0.0001"
              value={formData.learningRate}
              onChange={(e) => setFormData({ ...formData, learningRate: parseFloat(e.target.value) })}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Batch Size</label>
            <input
              type="number"
              className="form-input"
              min="1"
              max="1024"
              value={formData.batchSize}
              onChange={(e) => setFormData({ ...formData, batchSize: parseInt(e.target.value) })}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Epochs</label>
            <input
              type="number"
              className="form-input"
              min="1"
              max="10000"
              value={formData.epochs}
              onChange={(e) => setFormData({ ...formData, epochs: parseInt(e.target.value) })}
            />
          </div>
        </div>

        {selectedDataset && selectedNetwork && (
          <div className="message message-info">
            Dataset: {selectedDataset.name} | Network: {selectedNetwork.name}
          </div>
        )}

        <button
          type="submit"
          className="btn btn-primary"
          disabled={!selectedDataset || !selectedNetwork || submitting}
          style={{ marginTop: '1rem' }}
        >
          {submitting ? <><span className="loader"></span> Starting...</> : 'Start Training'}
        </button>
      </form>

      {/* Training Runs List */}
      <h3 style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>Training History</h3>

      {loading ? (
        <div className="empty-state">
          <span className="loader"></span>
          <p>Loading training runs...</p>
        </div>
      ) : trainingRuns.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">🏋️</div>
          <p>No training runs yet</p>
        </div>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Status</th>
              <th>LR</th>
              <th>Batch</th>
              <th>Epochs</th>
              <th>Best Accuracy</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {trainingRuns.map((run) => (
              <tr key={run.id}>
                <td>{run.id}</td>
                <td>{getStatusBadge(run.status)}</td>
                <td>{run.learning_rate}</td>
                <td>{run.batch_size}</td>
                <td>{run.epochs}</td>
                <td>{run.best_accuracy ? `${(run.best_accuracy * 100).toFixed(2)}%` : '-'}</td>
                <td>
                  <div style={{ display: 'flex', gap: '0.25rem' }}>
                    {run.status === 'completed' && (
                      <button
                        className="btn btn-success"
                        onClick={() => handleViewResults(run)}
                      >
                        View
                      </button>
                    )}
                    {(run.status === 'pending' || run.status === 'running') && (
                      <button
                        className="btn btn-secondary"
                        onClick={() => handleCancel(run.id)}
                      >
                        Cancel
                      </button>
                    )}
                    <button
                      className="btn btn-danger"
                      onClick={() => handleDelete(run.id)}
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default TrainingPanel;
