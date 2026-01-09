import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Datasets API
export const datasetsApi = {
  upload: async (file, name) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    const response = await api.post('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  list: async () => {
    const response = await api.get('/datasets/');
    return response.data;
  },

  get: async (id) => {
    const response = await api.get(`/datasets/${id}`);
    return response.data;
  },

  delete: async (id) => {
    const response = await api.delete(`/datasets/${id}`);
    return response.data;
  },

  preview: async (id, rows = 10) => {
    const response = await api.get(`/datasets/${id}/preview?rows=${rows}`);
    return response.data;
  },
};

// Networks API
export const networksApi = {
  create: async (data) => {
    const response = await api.post('/networks/', data);
    return response.data;
  },

  list: async () => {
    const response = await api.get('/networks/');
    return response.data;
  },

  get: async (id) => {
    const response = await api.get(`/networks/${id}`);
    return response.data;
  },

  update: async (id, data) => {
    const response = await api.put(`/networks/${id}`, data);
    return response.data;
  },

  delete: async (id) => {
    const response = await api.delete(`/networks/${id}`);
    return response.data;
  },

  suggest: async (datasetId, taskDescription = null) => {
    const response = await api.post('/networks/suggest', {
      dataset_id: datasetId,
      task_description: taskDescription,
    });
    return response.data;
  },
};

// Training API
export const trainingApi = {
  create: async (data) => {
    const response = await api.post('/training/', data);
    return response.data;
  },

  list: async (datasetId = null) => {
    const params = datasetId ? `?dataset_id=${datasetId}` : '';
    const response = await api.get(`/training/${params}`);
    return response.data;
  },

  get: async (id) => {
    const response = await api.get(`/training/${id}`);
    return response.data;
  },

  cancel: async (id) => {
    const response = await api.post(`/training/${id}/cancel`);
    return response.data;
  },

  delete: async (id) => {
    const response = await api.delete(`/training/${id}`);
    return response.data;
  },
};

// Optimization API
export const optimizationApi = {
  create: async (data) => {
    const response = await api.post('/optimization/', data);
    return response.data;
  },

  list: async (datasetId = null, algorithm = null) => {
    const params = new URLSearchParams();
    if (datasetId) params.append('dataset_id', datasetId);
    if (algorithm) params.append('algorithm', algorithm);
    const queryString = params.toString() ? `?${params.toString()}` : '';
    const response = await api.get(`/optimization/${queryString}`);
    return response.data;
  },

  get: async (id) => {
    const response = await api.get(`/optimization/${id}`);
    return response.data;
  },

  cancel: async (id) => {
    const response = await api.post(`/optimization/${id}/cancel`);
    return response.data;
  },

  delete: async (id) => {
    const response = await api.delete(`/optimization/${id}`);
    return response.data;
  },
};

// Health API
export const healthApi = {
  check: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;
