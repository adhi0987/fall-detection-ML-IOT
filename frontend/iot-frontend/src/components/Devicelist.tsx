import { useEffect, useState } from "react";

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
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4">Unique Devices</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left table-auto">
          <thead className="bg-gray-200">
            <tr>
              <th className="p-3">MAC Address</th>
              <th className="p-3">Action</th>
            </tr>
          </thead>
          <tbody>
            {devices.length > 0 ? (
              devices.map((mac) => (
                <tr
                  key={mac}
                  className={`border-b hover:bg-gray-100 ${selectedMac === mac ? 'bg-blue-100' : ''}`}
                >
                  <td className="p-3 font-mono">{mac}</td>
                  <td className="p-3">
                    <button
                      onClick={() => set_mac_addr(mac)}
                      className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors"
                    >
                      View Analytics
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={2} className="p-3 text-center text-gray-500">No devices found.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
export default DeviceList;