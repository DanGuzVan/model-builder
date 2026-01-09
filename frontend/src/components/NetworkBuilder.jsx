import { useState, useEffect } from 'react';
import { networksApi } from '../api';

function NetworkBuilder({ selectedDataset, selectedNetwork, onSelectNetwork }) {
  const [networks, setNetworks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [suggesting, setSuggesting] = useState(false);

  const [formData, setFormData] = useState({
    name: '',
    hiddenLayers: [64, 32],
    dropout: 0.2,
  });

  useEffect(() => {
    loadNetworks();
  }, []);

  const loadNetworks = async () => {
    try {
      setLoading(true);
      const response = await networksApi.list();
      setNetworks(response.networks || []);
      setError(null);
    } catch (err) {
      setError('Failed to load networks');
    } finally {
      setLoading(false);
    }
  };

  const handleLayerChange = (index, value) => {
    const newLayers = [...formData.hiddenLayers];
    newLayers[index] = parseInt(value) || 0;
    setFormData({ ...formData, hiddenLayers: newLayers });
  };

  const addLayer = () => {
    setFormData({
      ...formData,
      hiddenLayers: [...formData.hiddenLayers, 32],
    });
  };

  const removeLayer = (index) => {
    if (formData.hiddenLayers.length <= 1) return;
    const newLayers = formData.hiddenLayers.filter((_, i) => i !== index);
    setFormData({ ...formData, hiddenLayers: newLayers });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }

    if (!formData.name.trim()) {
      setError('Please enter a network name');
      return;
    }

    try {
      const networkConfig = {
        name: formData.name,
        config: {
          input_size: selectedDataset.num_features,
          hidden_layers: formData.hiddenLayers.filter((l) => l > 0),
          output_size: selectedDataset.num_classes,
          dropout: formData.dropout,
        },
      };

      await networksApi.create(networkConfig);
      setFormData({ name: '', hiddenLayers: [64, 32], dropout: 0.2 });
      await loadNetworks();
      setError(null);
    } catch (err) {
      setError('Failed to create network');
    }
  };

  const handleSuggest = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }

    try {
      setSuggesting(true);
      setError(null);
      const suggestion = await networksApi.suggest(selectedDataset.id);
      setFormData({
        name: `Suggested Network for ${selectedDataset.name}`,
        hiddenLayers: suggestion.suggested_config.hidden_layers,
        dropout: suggestion.suggested_config.dropout,
      });
    } catch (err) {
      setError('Failed to get network suggestion. Make sure Ollama is running.');
    } finally {
      setSuggesting(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this network?')) return;

    try {
      await networksApi.delete(id);
      if (selectedNetwork?.id === id) {
        onSelectNetwork(null);
      }
      await loadNetworks();
    } catch (err) {
      setError('Failed to delete network');
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Network Builder</h2>
        <button className="btn btn-secondary" onClick={loadNetworks}>
          Refresh
        </button>
      </div>

      {error && <div className="message message-error">{error}</div>}

      {!selectedDataset && (
        <div className="message message-info">
          Please select a dataset first to build a network with the correct input/output sizes.
        </div>
      )}

      {/* Network Builder Form */}
      <form onSubmit={handleSubmit} className="panel" style={{ background: '#f9f9f9' }}>
        <h3 style={{ marginBottom: '1rem' }}>Create New Network</h3>

        <div className="form-group">
          <label className="form-label">Network Name</label>
          <input
            type="text"
            className="form-input"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            placeholder="Enter network name"
          />
        </div>

        <div className="form-group">
          <label className="form-label">Hidden Layers</label>
          <div className="layer-list">
            {formData.hiddenLayers.map((layer, index) => (
              <div key={index} className="layer-item">
                <span>Layer {index + 1}:</span>
                <input
                  type="number"
                  min="1"
                  max="1024"
                  value={layer}
                  onChange={(e) => handleLayerChange(index, e.target.value)}
                />
                <span>neurons</span>
                {formData.hiddenLayers.length > 1 && (
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={() => removeLayer(index)}
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
          </div>
          <button type="button" className="btn btn-secondary" onClick={addLayer}>
            Add Layer
          </button>
        </div>

        <div className="form-group">
          <label className="form-label">Dropout Rate: {formData.dropout}</label>
          <input
            type="range"
            min="0"
            max="0.5"
            step="0.05"
            value={formData.dropout}
            onChange={(e) => setFormData({ ...formData, dropout: parseFloat(e.target.value) })}
            style={{ width: '100%' }}
          />
        </div>

        {selectedDataset && (
          <div className="message message-info">
            Input Size: {selectedDataset.num_features} | Output Size: {selectedDataset.num_classes}
          </div>
        )}

        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
          <button type="submit" className="btn btn-primary" disabled={!selectedDataset}>
            Create Network
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleSuggest}
            disabled={!selectedDataset || suggesting}
          >
            {suggesting ? <><span className="loader"></span> Suggesting...</> : 'LLM Suggest'}
          </button>
        </div>
      </form>

      {/* Network List */}
      <h3 style={{ marginTop: '1.5rem', marginBottom: '1rem' }}>Saved Networks</h3>

      {loading ? (
        <div className="empty-state">
          <span className="loader"></span>
          <p>Loading networks...</p>
        </div>
      ) : networks.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">🧠</div>
          <p>No networks created yet</p>
        </div>
      ) : (
        <div className="grid grid-2">
          {networks.map((network) => (
            <div
              key={network.id}
              className={`card ${selectedNetwork?.id === network.id ? 'selected' : ''}`}
              onClick={() => onSelectNetwork(network)}
            >
              <div className="card-title">{network.name}</div>
              <div className="card-meta">
                <p>Input: {network.config.input_size}</p>
                <p>Hidden: [{network.config.hidden_layers.join(', ')}]</p>
                <p>Output: {network.config.output_size}</p>
                <p>Dropout: {network.config.dropout}</p>
              </div>
              <button
                className="btn btn-danger"
                style={{ marginTop: '0.5rem' }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(network.id);
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default NetworkBuilder;
