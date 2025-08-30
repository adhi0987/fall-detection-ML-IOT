import React, { useState, useEffect } from 'react';

function DeviceList() {
  const [devices, setDevices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDevices = async () => {
      try {
        const response = await fetch('https://fall-prediction-api.onrender.com/getdevices');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setDevices(data.unique_devices);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDevices();
  }, []); // The empty array ensures this effect runs only once on component mount

  if (loading) {
    return <div>Loading devices...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div>
      <h1>Unique Devices</h1>
      <ul>
        {devices.length > 0 ? (
          devices.map((mac, index) => (
            <li key={index}>{mac}</li>
          ))
        ) : (
          <li>No devices found.</li>
        )}
      </ul>
    </div>
  );
}

export default DeviceList;