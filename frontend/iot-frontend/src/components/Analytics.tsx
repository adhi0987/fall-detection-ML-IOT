import React, { useEffect, useState } from 'react';
import { ScriptableContext } from 'chart.js';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import "../../styles/analytics.styles.css";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Define the shape of the data point object
interface DataPoint {
  id: number;
  mac_addr: string;
  max_Ax: number;
  min_Ax: number;
  var_Ax: number;
  mean_Ax: number;
  max_Ay: number;
  min_Ay: number;
  var_Ay: number;
  mean_Ay: number;
  max_Az: number;
  min_Az: number;
  var_Az: number;
  mean_Az: number;
  max_pitch: number;
  min_pitch: number;
  var_pitch: number;
  mean_pitch: number;
  prediction: number;
  prediction_label: string;
  timestamp: string;
}

interface AnalyticsProps {
  macid: string | null;
}

const Analytics: React.FC<AnalyticsProps> = ({ macid }) => {
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

useEffect(() => {
  if (!macid) return;

  let intervalId: ReturnType<typeof setInterval>;

  const fetchDataPoints = async () => {
    try {
      const response = await fetch(`https://fall-prediction-api.onrender.com/getdatapoints/${macid}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setDataPoints(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "An unknown error occurred.");
    } finally {
      setLoading(false);
    }
  };

  setLoading(true);
  fetchDataPoints(); // initial load
  intervalId = setInterval(fetchDataPoints, 5000); // refresh every 5s

  return () => clearInterval(intervalId); // cleanup
}, [macid]);


  if (!macid) {
    return <div className="text-center p-8 text-gray-500">Select a device from the list to view its analytics.</div>;
  }

  if (loading) {
    return <div className="text-center p-8">Loading analytics for {macid}...</div>;
  }

  if (error) {
    return <div className="text-center p-8 text-red-500">Error: {error}</div>;
  }

  // Prediction and accelerometer chart data preparation
  const labels = dataPoints.map((data) => new Date(data.timestamp).toLocaleString());
  const predictionData = dataPoints.map((data) => Number(data.prediction));
  const maxAxData = dataPoints.map((data) => data.max_Ax);
  const minAxData = dataPoints.map((data) => data.min_Ax);
  const meanAxData = dataPoints.map((data) => data.mean_Ax);

  const maxAyData = dataPoints.map((data) => data.max_Ay);
  const minAyData = dataPoints.map((data) => data.min_Ay);
  const meanAyData = dataPoints.map((data) => data.mean_Ay);

  const maxAzData = dataPoints.map((data) => data.max_Az);
  const minAzData = dataPoints.map((data) => data.min_Az);
  const meanAzData = dataPoints.map((data) => data.mean_Az);

  const maxPData = dataPoints.map((data) => data.max_pitch);
  const minPData = dataPoints.map((data) => data.min_pitch);
  const meanPData = dataPoints.map((data) => data.mean_pitch);

  const predictionChartData = {
    labels,
    datasets: [
      {
        label: 'Fall Prediction',
        data: predictionData,
        pointBackgroundColor: (context: ScriptableContext<'line'>) => {
          const value = context.dataset.data[context.dataIndex] as number;
          return value === 1 ? 'rgb(255, 99, 132)' : 'rgb(53, 162, 235)';
        },
        pointBorderColor: (context: ScriptableContext<'line'>) => {
          const value = context.dataset.data[context.dataIndex] as number;
          return value === 1 ? 'rgb(255, 99, 132)' : 'rgb(53, 162, 235)';
        },
        borderWidth: 1,
        showLine: false,   // Show only points, no connecting line
        pointRadius: 6,    // Makes points visible
        pointHoverRadius: 8,
      },
    ],
  };

  const accelerometerChartDataX = {
    labels,
    datasets: [
      {
        label: 'Max X-axis Acceleration',
        data: maxAxData,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.5,
      },
      {
        label: 'Min X-axis Acceleration',
        data: minAxData,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.5,
      },
      {
        label: 'Mean X-axis Acceleration',
        data: meanAxData,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.5,
      },
    ],
  };

  const accelerometerChartDataY = {
    labels,
    datasets: [
      {
        label: 'Max Y-axis Acceleration',
        data: maxAyData,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.5,
      },
      {
        label: 'Min Y-axis Acceleration',
        data: minAyData,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.5,
      },
      {
        label: 'Mean Y-axis Acceleration',
        data: meanAyData,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.5,
      },
    ],
  };

  const accelerometerChartDataZ = {
    labels,
    datasets: [
      {
        label: 'Max Z-axis Acceleration',
        data: maxAzData,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.5,
      },
      {
        label: 'Min Z-axis Acceleration',
        data: minAzData,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.5,
      },
      {
        label: 'Mean Z-axis Acceleration',
        data: meanAzData,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.5,
      },
    ],
  };

  const accelerometerChartDataP = {
    labels,
    datasets: [
      {
        label: 'Max Pitch Acceleration',
        data: maxPData,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.5,
      },
      {
        label: 'Min Pitch Acceleration',
        data: minPData,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.5,
      },
      {
        label: 'Mean Pitch Acceleration',
        data: meanPData,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.5,
      },
    ],
  };

  const predictionOptions = {
   responsive: true, 
   
    plugins: {
      legend: { position: 'top' as const },
      title: { display: true, text: 'Fall Prediction Events' },
    },
    // Uncomment if y-axis ticks and labels needed
    // scales: {
    //   y: {
    //     ticks: {
    //       callback: (value: number) => (value === 1 ? 'Fall' : 'No Fall'),
    //     },
    //     min: 0,
    //     max: 1,
    //     stepSize: 1,
    //     title: { display: true, text: 'Prediction' },
    //   },
    //   x: { title: { display: true, text: 'Time' } },
    // },
  };

  const accelerometerOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' as const },
      title: { display: true, text: 'Accelerometer Data (X-axis) over Time' },
    },
    scales: {
      x: { title: { display: true, text: 'Time' } },
      y: { title: { display: true, text: 'Acceleration (g)' } },
    },
  };

  return (
    <div className="analytics-container">
      <h2>Analytics for MAC: {macid}</h2>

      {dataPoints.length > 0 ? (
        <>
          <div className="charts-grid">
            <div className="chart-card">
              <h3>Accelerometer Data (X-axis)</h3>
              <Line data={accelerometerChartDataX} options={accelerometerOptions} />
            </div>
            <div className="chart-card">
              <h3>Accelerometer Data (Y-axis)</h3>
              <Line data={accelerometerChartDataY} options={accelerometerOptions} />
            </div>
            <div className="chart-card">
              <h3>Accelerometer Data (Z-axis)</h3>
              <Line data={accelerometerChartDataZ} options={accelerometerOptions} />
            </div>
            <div className="chart-card">
              <h3>Accelerometer Data (Pitch)</h3>
              <Line data={accelerometerChartDataP} options={accelerometerOptions} />
            </div>
          </div>
          <div className="full-width-chart">
            <div className="chart-card">
              <h3>Fall Prediction</h3>
              <Line data={predictionChartData} options={predictionOptions} />
            </div>
          </div>
          {/* Raw Data Table */}
          <h3>Raw Data Points</h3>
          <div className="table-responsive">
            <table className="data-table">
              <thead className="bg-gray-200">
                <tr>
                  <th className="p-2">Sno</th>
                  <th className="p-2">Timestamp</th>
                  <th className="p-2">Prediction</th>
                  <th className="p-2">max_Ax</th>
                  <th className="p-2">min_Ax</th>
                  <th className="p-2">var_Ax</th>
                  <th className="p-2">mean_Ax</th>
                  <th className="p-2">max_Ay</th>
                  <th className="p-2">min_Ay</th>
                  <th className="p-2">var_Ay</th>
                  <th className="p-2">mean_Ay</th>
                  <th className="p-2">max_Az</th>
                  <th className="p-2">min_Az</th>
                  <th className="p-2">var_Az</th>
                  <th className="p-2">mean_Az</th>
                  <th className="p-2">max_pitch</th>
                  <th className="p-2">min_pitch</th>
                  <th className="p-2">var_pitch</th>
                  <th className="p-2">mean_pitch</th>
                </tr>
              </thead>
              <tbody>
                {dataPoints.map((dataPoint, index) => (
                  <tr key={dataPoint.id} className="border-b hover:bg-gray-100">
                    <td className="p-2">{index + 1}</td>
                    <td className="p-2">{new Date(dataPoint.timestamp).toLocaleString()}</td>
                    <td className="p-2 font-medium">{dataPoint.prediction_label}</td>
                    <td className="p-2">{dataPoint.max_Ax.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.min_Ax.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.var_Ax.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.mean_Ax.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.max_Ay.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.min_Ay.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.var_Ay.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.mean_Ay.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.max_Az.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.min_Az.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.var_Az.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.mean_Az.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.max_pitch.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.min_pitch.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.var_pitch.toFixed(4)}</td>
                    <td className="p-2">{dataPoint.mean_pitch.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="text-center p-8 text-gray-500">No data points to display for this device.</div>
      )}
    </div>
  );
};

export default Analytics;