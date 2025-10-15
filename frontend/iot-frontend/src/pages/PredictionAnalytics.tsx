import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// --- API Configuration ---
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ; 

// --- Data Type Definition ---
interface PredictionData {
  id: number;
  mac_addr: string;
  prediction: number;
  prediction_label: string;
  timestamp: string;
  max_Ax: number;
  mean_Ax: number;
  max_Ay: number;
  mean_Ay: number;
  max_Az: number;
  mean_Az: number;
}

const PredictionAnalytics: React.FC = () => {
  const [data, setData] = useState<PredictionData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/fetchdata?source_type=predicted`);
        setData(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch prediction data. Is the backend server running?');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh data every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const chartData = data.map(item => ({
    time: new Date(item.timestamp).toLocaleTimeString(),
    Fall: item.prediction === 1 ? 1 : 0,
    'No Fall': item.prediction === 0 ? 1 : 0,
  })).reverse();


  if (loading) return <div className="info-message">Loading prediction data...</div>;
  if (error) return <div className="error-message">{error}</div>;
  if (data.length === 0) return <div className="info-message">Sorry, no prediction data is present yet.</div>;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Prediction Analytics</h1>
      </div>

      <div className="chart-container">
        <h2 className="section-title">Real-time Fall Detection Events</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="Fall" stroke="#e53e3e" strokeWidth={2} activeDot={{ r: 8 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <h2 className="section-title">Prediction Data Log</h2>
      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Record Id</th>
              <th>Device MAC</th>
              <th>Avg Ax</th>
              <th>Avg Ay</th>
              <th>Avg Az</th>
              <th>Prediction</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item) => (
              <tr key={item.id}>
                {/* <td>{new Date(item.timestamp).toLocaleString()}</td> */}
                <td>{item.id}</td>
                <td>{item.mac_addr}</td>
                <td>{item.mean_Ax.toFixed(2)}</td>
                <td>{item.mean_Ay.toFixed(2)}</td>
                <td>{item.mean_Az.toFixed(2)}</td>
                <td>
                  <span className={item.prediction === 1 ? 'fall' : 'no-fall'}>
                    {item.prediction_label}
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

export default PredictionAnalytics;
