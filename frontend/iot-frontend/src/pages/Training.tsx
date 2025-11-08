import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// --- API Configuration ---
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ;

// --- Data Type Definition ---
interface TrainingData {
  id: number;
  mac_addr: string;
  prediction: number;
  timestamp: string;
  max_Ax: number;
  mean_Ax: number;
  max_Ay: number;
  mean_Ay: number;
  max_Az: number;
  mean_Az: number;
  source_type?:string;
}

const Training: React.FC = () => {
  const [unlabelledData, setUnlabelledData] = useState<TrainingData[]>([]);
  const [labelledData, setLabelledData] = useState<TrainingData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<string>('');

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      // Fetch unlabelled data (where prediction is -1)
      const unlabelledRes = await axios.get(`${API_BASE_URL}/fetchdata?source_type=labelled`);
      setUnlabelledData(unlabelledRes.data.filter((item: TrainingData) => item.prediction === -1));

      // Fetch the full labeled dataset (for display)
      const labelledRes = await axios.get(`${API_BASE_URL}/showdataset`);
      setLabelledData(labelledRes.data);

      setError(null);
    } catch (err) {
      setError('Failed to fetch training data. Is the backend server running?');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // ---Tagging Function ---
  const handleTagging = async (id: number, prediction: number) => {
    
    const itemToLabel = unlabelledData.find((item) => item.id === id);
    if (!itemToLabel) return; 

    try {
      // 1. Make the API call to update the backend
      await axios.put(`${API_BASE_URL}/labeldata/${id}`, { prediction });

      // 2. On success, update the local state without re-fetching
      
      // Remove the item from the unlabelledData array
      setUnlabelledData(currentUnlabelled =>
        currentUnlabelled.filter(item => item.id !== id)
      );

      // Add the (now labelled) item to the labelledData array
      const newlyLabelledItem = { ...itemToLabel, prediction: prediction };
      setLabelledData(currentLabelled =>
        [newlyLabelledItem, ...currentLabelled] // Add to the beginning of the list
      );

    } catch (err) {
      alert('Failed to label data point. The change will not be saved.');
      console.error(err);
    }
  };

  const handleTrainModel = async () => {
    setTrainingStatus('Training in progress... This may take a few minutes.');
    try {
      const response = await axios.post(`${API_BASE_URL}/trainmodel`);
      setTrainingStatus(`Training complete! ${response.data.message} (Dataset Size: ${response.data.dataset_size})`);
    } catch (err: any) {
      setTrainingStatus(`Training failed: ${err.response?.data?.detail || 'An unknown error occurred.'}`);
      console.error(err);
    }
  };
  
  if (loading) return <div className="info-message">Loading training data...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Model Training & Data Labeling</h1>
        <button className="action-button primary-button" onClick={handleTrainModel} disabled={trainingStatus.includes('progress')}>
          Train New Model
        </button>
      </div>
      {trainingStatus && <div className="info-message">{trainingStatus}</div>}
      
      <h2 className="section-title">Data Points to Label</h2>
      {unlabelledData.length > 0 ? (
        <div className="data-table-container">
          <table className="data-table">
            {/* ... table head ... */}
            <thead>
                <tr>
                    <th>Record Id</th>
                    <th>Timestamp</th>
                    <th>Avg Ax</th>
                    <th>Avg Ay</th>
                    <th>Avg Az</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
              {unlabelledData.map((item) => (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td>{new Date(item.timestamp).toLocaleString()}</td>
                  <td>{item.mean_Ax.toFixed(2)}</td>
                  <td>{item.mean_Ay.toFixed(2)}</td>
                  <td>{item.mean_Az.toFixed(2)}</td>
                  <td>
                    <select className="tag-select" defaultValue="" onChange={(e) => handleTagging(item.id, parseInt(e.target.value))}>
                      <option value="" disabled>Select Label</option>
                      <option value="0">No Fall</option>
                      <option value="1">Fall</option>
                    </select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>No new data points to label. Great job!</p>
      )}

      <h2 className="section-title">Current Labeled Dataset ({labelledData.length} records)</h2>
      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
                <th>Record Id</th>
                {/* <th>Device MAC</th> */}
                <th>timestamp</th>
                <th>Avg Ax</th>
                <th>Avg Ay</th>
                <th>Avg Az</th>
                <th>Source Type</th>
                <th>Label</th>
            </tr>
          </thead>
          <tbody>
            {labelledData.map((item) => (
              <tr key={item.id}>
                <td>{item.id}</td>
                <td>{new Date(item.timestamp).toLocaleString()}</td>
                {/* <td>{item.mac_addr}</td> */}
                <td>{item.mean_Ax.toFixed(2)}</td>
                <td>{item.mean_Ay.toFixed(2)}</td>
                <td>{item.mean_Az.toFixed(2)}</td>
                <td>{item.source_type} </td>
                <td>
                  <span className={item.prediction === 1 ? 'fall' : 'no-fall'}>
                    {item.prediction === 1 ? 'Fall' : 'No Fall'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Training;