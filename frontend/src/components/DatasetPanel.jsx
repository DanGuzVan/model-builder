import { useState, useEffect } from 'react';
import { datasetsApi } from '../api';

function DatasetPanel({ selectedDataset, onSelectDataset }) {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadName, setUploadName] = useState('');
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await datasetsApi.list();
      setDatasets(response.datasets || []);
      setError(null);
    } catch (err) {
      setError('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!uploadName.trim()) {
      setError('Please enter a dataset name');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      await datasetsApi.upload(file, uploadName);
      setUploadName('');
      event.target.value = '';
      await loadDatasets();
    } catch (err) {
      setError('Failed to upload dataset');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;

    try {
      await datasetsApi.delete(id);
      if (selectedDataset?.id === id) {
        onSelectDataset(null);
      }
      await loadDatasets();
    } catch (err) {
      setError('Failed to delete dataset');
    }
  };

  const handlePreview = async (dataset) => {
    try {
      const data = await datasetsApi.preview(dataset.id);
      setPreview({ ...data, name: dataset.name });
    } catch (err) {
      setError('Failed to load preview');
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Datasets</h2>
        <button className="btn btn-secondary" onClick={loadDatasets}>
          Refresh
        </button>
      </div>

      {error && <div className="message message-error">{error}</div>}

      {/* Upload Section */}
      <div className="panel" style={{ background: '#f9f9f9' }}>
        <h3 style={{ marginBottom: '1rem' }}>Upload Dataset</h3>
        <div className="form-group">
          <label className="form-label">Dataset Name</label>
          <input
            type="text"
            className="form-input"
            value={uploadName}
            onChange={(e) => setUploadName(e.target.value)}
            placeholder="Enter dataset name"
          />
        </div>
        <label className="file-upload">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            disabled={uploading}
          />
          {uploading ? (
            <span><span className="loader"></span> Uploading...</span>
          ) : (
            <span>Click to upload CSV file</span>
          )}
        </label>
      </div>

      {/* Dataset List */}
      {loading ? (
        <div className="empty-state">
          <span className="loader"></span>
          <p>Loading datasets...</p>
        </div>
      ) : datasets.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">📊</div>
          <p>No datasets uploaded yet</p>
        </div>
      ) : (
        <div className="grid grid-2">
          {datasets.map((dataset) => (
            <div
              key={dataset.id}
              className={`card ${selectedDataset?.id === dataset.id ? 'selected' : ''}`}
              onClick={() => onSelectDataset(dataset)}
            >
              <div className="card-title">{dataset.name}</div>
              <div className="card-meta">
                <p>Features: {dataset.num_features}</p>
                <p>Classes: {dataset.num_classes}</p>
                <p>Samples: {dataset.num_samples.toLocaleString()}</p>
              </div>
              <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem' }}>
                <button
                  className="btn btn-secondary"
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePreview(dataset);
                  }}
                >
                  Preview
                </button>
                <button
                  className="btn btn-danger"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(dataset.id);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Preview Modal */}
      {preview && (
        <div className="panel" style={{ marginTop: '1rem' }}>
          <div className="panel-header">
            <h3 className="panel-title">Preview: {preview.name}</h3>
            <button className="btn btn-secondary" onClick={() => setPreview(null)}>
              Close
            </button>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  {preview.columns.map((col, i) => (
                    <th key={i}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.data.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td key={j}>{typeof cell === 'number' ? cell.toFixed(4) : cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default DatasetPanel;
