import { useEffect, useState } from "react";

interface Props {
    macid: string;
}

// Define the shape of the data point object
interface DataPoint {
    id: number;
    mac_addr: string;
    max_Ax: number;
    // ... add all other properties as they appear in the API response
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

function Analytics({ macid }: Props) {
    // The state now correctly holds an array of DataPoint objects
    const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
    const [loading, setLoading] = useState(false); // Set to false initially, fetch when macid is available
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        // Only fetch data if macid is a valid, non-empty string
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
                // The API returns an array of objects directly
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
    }, [macid]); // Re-run the effect whenever macid changes

    if (!macid) {
        return <div>Select a device to view analytics.</div>;
    }

    if (loading) {
        return <div>Loading data points...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    return (
        <div>
            <h2>Analytics for {macid}</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>Prediction</th>
                        <th>max_Ax</th>
                        {/* ... add all other table headers ... */}
                        <th>min_Ax</th>
                        <th>var_Ax</th>
                        <th>mean_Ax</th>
                        <th>max_Ay</th>
                        <th>min_Ay</th>
                        <th>var_Ay</th>
                        <th>mean_Ay</th>
                        <th>max_Az</th>
                        <th>min_Az</th>
                        <th>var_Az</th>
                        <th>mean_Az</th>
                        <th>max_pitch</th>
                        <th>min_pitch</th>
                        <th>var_pitch</th>
                        <th>mean_pitch</th>
                    </tr>
                </thead>
                <tbody>
                    {dataPoints.length > 0 ? (
                        dataPoints.map((dataPoint) => (
                            <tr key={dataPoint.id}>
                                <td>{dataPoint.id}</td>
                                <td>{new Date(dataPoint.timestamp).toLocaleString()}</td>
                                <td>{dataPoint.prediction_label}</td>
                                <td>{dataPoint.max_Ax.toFixed(2)}</td>
                                {/* ... add all other data cells for each dataPoint */}
                                <td>{dataPoint.min_Ax.toFixed(2)}</td>
                                <td>{dataPoint.var_Ax.toFixed(2)}</td>
                                <td>{dataPoint.mean_Ax.toFixed(2)}</td>
                                <td>{dataPoint.max_Ay.toFixed(2)}</td>
                                <td>{dataPoint.min_Ay.toFixed(2)}</td>
                                <td>{dataPoint.var_Ay.toFixed(2)}</td>
                                <td>{dataPoint.mean_Ay.toFixed(2)}</td>
                                <td>{dataPoint.max_Az.toFixed(2)}</td>
                                <td>{dataPoint.min_Az.toFixed(2)}</td>
                                <td>{dataPoint.var_Az.toFixed(2)}</td>
                                <td>{dataPoint.mean_Az.toFixed(2)}</td>
                                <td>{dataPoint.max_pitch.toFixed(2)}</td>
                                <td>{dataPoint.min_pitch.toFixed(2)}</td>
                                <td>{dataPoint.var_pitch.toFixed(2)}</td>
                                <td>{dataPoint.mean_pitch.toFixed(2)}</td>
                            </tr>
                        ))
                    ) : (
                        <tr>
                            <td colSpan={19}>No data found for this device.</td>
                        </tr>
                    )}
                </tbody>
            </table>
        </div>
    );
}

export default Analytics;