import { useState } from "react";
import Navbar from "./components/Navbar";
import DeviceList from "./components/Devicelist";
import Analytics from "./components/Analytics";

const App = () => {
  const [macaddr, setMacAddr] = useState<string | null>(null);

  return (
    <div className="bg-gray-100 min-h-screen font-inter">
      <Navbar />
      <div className="container mx-auto px-4 py-24">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="col-span-1">
            <DeviceList set_mac_addr={setMacAddr} selectedMac={macaddr} />
          </div>
          <div className="col-span-1 md:col-span-2">
            <Analytics macid={macaddr} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;