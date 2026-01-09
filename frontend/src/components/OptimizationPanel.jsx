import { useState, useEffect } from 'react';
import { optimizationApi } from '../api';

function OptimizationPanel({ selectedDataset, onOptimizationComplete }) {
  const [optimizationRuns, setOptimizationRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [algorithm, setAlgorithm] = useState('pso');

  const [psoConfig, setPsoConfig] = useState({
    num_particles: 30,
    max_iterations: 100,
    w: 0.7,
    c1: 1.5,
    c2: 1.5,
  });

  const [gaConfig, setGaConfig] = useState({
    population_size: 50,
    generations: 100,
    mutation_rate: 0.1,
    crossover_rate: 0.8,
  });

  useEffect(() => {
    loadOptimizationRuns();
    const interval = setInterval(loadOptimizationRuns, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadOptimizationRuns = async () => {
    try {
      setLoading(true);
      const response = await optimizationApi.list();
      setOptimizationRuns(response.optimization_runs || []);
      setError(null);
    } catch (err) {
      setError('Failed to load optimization runs');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }

    try {
      setSubmitting(true);
      setError(null);

      const runData = {
        dataset_id: selectedDataset.id,
        algorithm: algorithm,
        config: algorithm === 'pso' ? psoConfig : gaConfig,
      };

      await optimizationApi.create(runData);
      await loadOptimizationRuns();
    } catch (err) {
      setError('Failed to start optimization');
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = async (id) => {
    try {
      await optimizationApi.cancel(id);
      await loadOptimizationRuns();
    } catch (err) {
      setError('Failed to cancel optimization');
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this optimization run?')) return;

    try {
      await optimizationApi.delete(id);
      await loadOptimizationRuns();
    } catch (err) {
      setError('Failed to delete optimization run');
    }
  };

  const handleViewResults = (run) => {
    onOptimizationComplete(run);
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
        <h2 className="panel-title">Optimization</h2>
        <button className="btn btn-secondary" onClick={loadOptimizationRuns}>
          Refresh
        </button>
      </div>

      {error && <div className="message message-error">{error}</div>}

      {!selectedDataset && (
        <div className="message message-info">
          Please select a dataset to run optimization.
        </div>
      )}

      {/* Optimization Form */}
      <form onSubmit={handleSubmit} className="panel" style={{ background: '#f9f9f9' }}>
        <h3 style={{ marginBottom: '1rem' }}>Start New Optimization</h3>

        <div className="form-group">
          <label className="form-label">Algorithm</label>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="radio"
                value="pso"
                checked={algorithm === 'pso'}
                onChange={(e) => setAlgorithm(e.target.value)}
              />
              Particle Swarm Optimization (PSO)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="radio"
                value="ga"
                checked={algorithm === 'ga'}
                onChange={(e) => setAlgorithm(e.target.value)}
              />
              Genetic Algorithm (GA)
            </label>
          </div>
        </div>

        {algorithm === 'pso' ? (
          <div className="grid grid-3">
            <div className="form-group">
              <label className="form-label">Particles</label>
              <input
                type="number"
                className="form-input"
                min="5"
                max="200"
                value={psoConfig.num_particles}
                onChange={(e) => setPsoConfig({ ...psoConfig, num_particles: parseInt(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Max Iterations</label>
              <input
                type="number"
                className="form-input"
                min="10"
                max="1000"
                value={psoConfig.max_iterations}
                onChange={(e) => setPsoConfig({ ...psoConfig, max_iterations: parseInt(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Inertia (w)</label>
              <input
                type="number"
                className="form-input"
                min="0"
                max="1"
                step="0.1"
                value={psoConfig.w}
                onChange={(e) => setPsoConfig({ ...psoConfig, w: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Cognitive (c1)</label>
              <input
                type="number"
                className="form-input"
                min="0"
                max="3"
                step="0.1"
                value={psoConfig.c1}
                onChange={(e) => setPsoConfig({ ...psoConfig, c1: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Social (c2)</label>
              <input
                type="number"
                className="form-input"
                min="0"
                max="3"
                step="0.1"
                value={psoConfig.c2}
                onChange={(e) => setPsoConfig({ ...psoConfig, c2: parseFloat(e.target.value) })}
              />
            </div>
          </div>
        ) : (
          <div className="grid grid-4">
            <div className="form-group">
              <label className="form-label">Population Size</label>
              <input
                type="number"
                className="form-input"
                min="10"
                max="500"
                value={gaConfig.population_size}
                onChange={(e) => setGaConfig({ ...gaConfig, population_size: parseInt(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Generations</label>
              <input
                type="number"
                className="form-input"
                min="10"
                max="1000"
                value={gaConfig.generations}
                onChange={(e) => setGaConfig({ ...gaConfig, generations: parseInt(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Mutation Rate</label>
              <input
                type="number"
                className="form-input"
                min="0"
                max="1"
                step="0.05"
                value={gaConfig.mutation_rate}
                onChange={(e) => setGaConfig({ ...gaConfig, mutation_rate: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Crossover Rate</label>
              <input
                type="number"
                className="form-input"
                min="0"
                max="1"
                step="0.05"
                value={gaConfig.crossover_rate}
                onChange={(e) => setGaConfig({ ...gaConfig, crossover_rate: parseFloat(e.target.value) })}
              />
            </div>
          </div>
        )}

        {selectedDataset && (
          <div className="message message-info">
            Dataset: {selectedDataset.name}
          </div>
        )}

        <button
          type="submit"
          className="btn btn-primary"
          disabled={!selectedDataset || submitting}
          style={{ marginTop: '1rem' }}
        >
          {submitting ? <><span className="loader"></span> Starting...</> : `Start ${algorithm.toUpperCase()} Optimization`}
        </button>
      </form>

      {/* Optimization Runs List */}
      <h3 style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>Optimization History</h3>

      {loading ? (
        <div className="empty-state">
          <span className="loader"></span>
          <p>Loading optimization runs...</p>
        </div>
      ) : optimizationRuns.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">🔬</div>
          <p>No optimization runs yet</p>
        </div>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Algorithm</th>
              <th>Status</th>
              <th>Best Accuracy</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {optimizationRuns.map((run) => (
              <tr key={run.id}>
                <td>{run.id}</td>
                <td>{run.algorithm.toUpperCase()}</td>
                <td>{getStatusBadge(run.status)}</td>
                <td>
                  {run.best_result
                    ? `${(run.best_result.accuracy * 100).toFixed(2)}%`
                    : '-'}
                </td>
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

export default OptimizationPanel;
