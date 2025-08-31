import { useEffect, useState } from "react";
import "../../styles/devicelist.styles.css";

interface DeviceListProps {
  set_mac_addr: (mac: string) => void;
  selectedMac: string | null;
}

const DeviceList: React.FC<DeviceListProps> = ({ set_mac_addr, selectedMac }) => {
  const [devices, setDevices] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDevices = async () => {
      try {
        const response = await fetch('https://fall-prediction-api.onrender.com/getdevices');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setDevices(data.unique_devices);
        if (data.unique_devices.length > 0) {
          set_mac_addr(data.unique_devices[0]);
        }
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

    fetchDevices();
  }, [set_mac_addr]);

  if (loading) {
    return <div className="text-center py-4">Loading devices...</div>;
  }

  if (error) {
    return <div className="text-center py-4 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="device-list-wrapper">
      <h2>Unique Devices</h2>
      <div className="device-table-container">
        <table className="device-table">
          <thead>
            <tr>
              <th>MAC Address</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {devices.length > 0 ? (
              devices.map((mac) => (
                <tr
                  key={mac}
                  className={selectedMac === mac ? 'selected' : ''}
                >
                  <td className="font-mono">{mac}</td>
                  <td>
                    <button
                      onClick={() => set_mac_addr(mac)}
                      className="btn-view"
                    >
                      View Analytics
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={2} style={{ textAlign: "center", color: "#6b7280", padding: "0.75rem 1rem" }}>
                  No devices found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DeviceList;