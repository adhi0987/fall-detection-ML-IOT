import React, { useEffect, useState } from 'react';
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
import { Line, Bar } from 'react-chartjs-2';

// Register the Chart.js components
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
    if (!macid) {
      return;
    }

    const fetchDataPoints = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`https://fall-prediction-api.onrender.com/getdatapoints/${macid}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setDataPoints(data);
      } catch (e: unknown) {
        if (e instanceof Error) {
          setError(e.message);
        } else {
          setError("An unknown error occurred.");
        }
      } finally {
        setLoading(false);
      }
    };
    fetchDataPoints();
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
  
  const labels = dataPoints.map((data) => new Date(data.timestamp).toLocaleString());
  const predictionData = dataPoints.map((data) => data.prediction);
  const maxAxData = dataPoints.map((data) => data.max_Ax);

  const predictionChartData = {
    labels,
    datasets: [
      {
        label: 'Fall Prediction',
        data: predictionData,
        backgroundColor: (context: { dataIndex: any; dataset: { data: { [x: string]: any; }; }; }) => {
            const index = context.dataIndex;
            const value = context.dataset.data[index];
            return value === 1 ? 'rgba(255, 99, 132, 0.5)' : 'rgba(53, 162, 235, 0.5)';
        },
        borderColor: (context: { dataIndex: any; dataset: { data: { [x: string]: any; }; }; }) => {
            const index = context.dataIndex;
            const value = context.dataset.data[index];
            return value === 1 ? 'rgb(255, 99, 132)' : 'rgb(53, 162, 235)';
        },
        borderWidth: 1,
      },
    ],
  };

  const accelerometerChartData = {
    labels,
    datasets: [
      {
        label: 'Max X-axis Acceleration',
        data: maxAxData,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.1,
      },
      {
        label: 'Min X-axis Acceleration',
        data: dataPoints.map((data) => data.min_Ax),
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        tension: 0.1,
      },
      {
        label: 'Mean X-axis Acceleration',
        data: dataPoints.map((data) => data.mean_Ax),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.1,
      },
    ],
  };

//   const predictionOptions = {
//     responsive: true,
//     plugins: {
//       legend: { position: 'top' as const },
//       title: { display: true, text: 'Fall Prediction Events' },
//     },
//     scales: {
//         y: {
//             // Set the y-axis type to 'category' to handle string labels
//             type: 'category',
//             labels: ['No Fall', 'Fall'],
//             title: { display: true, text: 'Prediction' },
//         },
//         x: {
//             title: { display: true, text: 'Time' },
//         },
//     }
//   };

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
    <div className="p-8 bg-white rounded-lg shadow-md mt-8">
      <h2 className="text-xl font-semibold mb-4">Analytics for MAC: {macid}</h2>
      
      {dataPoints.length > 0 ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div className="bg-gray-50 rounded-lg p-4 shadow">
              <h3 className="text-lg font-medium mb-2">Fall Prediction</h3>
              <Bar data={predictionChartData}  />
            </div>
            <div className="bg-gray-50 rounded-lg p-4 shadow">
              <h3 className="text-lg font-medium mb-2">Accelerometer Data (X-axis)</h3>
              <Line data={accelerometerChartData} options={accelerometerOptions} />
            </div>
          </div>

          <h3 className="text-xl font-semibold mb-4">Raw Data Points</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm table-auto">
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