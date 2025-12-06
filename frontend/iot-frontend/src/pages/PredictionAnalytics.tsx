// frontend/iot-frontend/src/pages/PredictionAnalytics.tsx

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// --- API Configuration ---
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL;

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

// --- Helper Functions ---

/**
 * Formats the Y-axis ticks to be more descriptive.
 * @param tick - The numerical value of the tick (e.g., 0 or 1).
 * @returns A string label for the tick.
 */
const yAxisTickFormatter = (tick: number) => {
    if (tick === 1) return 'Fall';
    if (tick === 0) return 'No Fall';
    return '';
};


const PredictionAnalytics: React.FC = () => {
  const [data, setData] = useState<PredictionData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // 1. Fetch initial data when the component mounts.
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/fetchdata?source_type=predicted`);
        
        // Filter initial data to only include the last 10 minutes.
        const tenMinutesAgo = Date.now() - 4*24*60 * 60 * 1000;
        const recentData = response.data.filter((item: PredictionData) => new Date(item.timestamp).getTime() > tenMinutesAgo);
        
        const sortedData = recentData.sort((a: PredictionData, b: PredictionData) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        
        setData(sortedData);
        setError(null);
      } catch (err) {
        setError('Failed to fetch prediction data. Is the backend server running?');
        console.error("Error fetching initial data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    // 2. Initialize the WebSocket connection.
    ws.current = new WebSocket(WEBSOCKET_URL);

    ws.current.onopen = () => {
        console.log(" WebSocket connection established");
    };

    ws.current.onmessage = (event) => {
        try {
            const newData = JSON.parse(event.data);
            // Append new data; the cleanup interval will handle removing old entries.
            setData((prevData) => [...prevData, newData]);
        } catch (error) {
            console.error("Failed to parse WebSocket message data:", error);
        }
    };
    
    ws.current.onerror = (event) => {
        console.error(" WebSocket error observed:", event);
        setError("WebSocket connection error. Real-time updates may not work.");
    };

    ws.current.onclose = (event) => {
        console.log("WebSocket connection closed.", event);
    };

    // 3. Set up an interval to periodically remove old data points.
    const cleanupInterval = setInterval(() => {
      setData(currentData => {
        const tenMinutesAgo = Date.now() - 10 * 60 * 1000;
        const freshData = currentData.filter(
          item => new Date(item.timestamp).getTime() > tenMinutesAgo
        );
        return freshData;
      });
    }, 10000); // Run cleanup every 10 seconds.


    // 4. Cleanup function for when the component unmounts.
    return () => {
        if (ws.current) {
            console.log("Closing WebSocket connection on component unmount.");
            ws.current.close();
        }
        // Clear the cleanup interval to prevent memory leaks.
        clearInterval(cleanupInterval);
    };
  }, []); // Empty dependency array ensures this effect runs only once on mount.

  // 5. Prepare data for the chart.
  const chartData = data.map(item => {
    const date = new Date(item.timestamp);
    return {
        time: !isNaN(date.getTime()) ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : "Invalid Time",
        Fall: item.prediction === 1 ? 1 : 0
    };
  });

  if (loading) return <div className="info-message">Loading prediction data...</div>;
  if (error) return <div className="error-message">{error}</div>;
  if (data.length === 0) return <div className="info-message">No prediction data from the last 10 minutes is available yet.</div>;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Prediction Analytics</h1>
        <p className="page-subtitle">Displaying real-time data from the last 10 minutes</p>
      </div>

      <div className="chart-container">
        <h2 className="section-title">Real-time Fall Detection Events</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis
              allowDecimals={false}
              domain={[0, 1]}
              ticks={[0, 1]}
              tickFormatter={yAxisTickFormatter}
            />
            <Tooltip
              formatter={(value: number) => {
                const label = value === 1 ? 'Fall Detected' : 'No Fall';
                return [label, 'Status'];
              }}
            />
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
            {/* Display data in reverse chronological order (newest first) */}
            {[...data].reverse().map((item) => (
              <tr key={item.id}>
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