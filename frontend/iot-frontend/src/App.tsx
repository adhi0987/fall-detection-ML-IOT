import './App.css'
import Navbar from './components/Navbar';
import DeviceList from './components/Devicelist';
import Analytics from './components/Analytics';
import { useState } from 'react';
function App() {
  const [macaddr,setMacAddr] = useState(String);
  return (
    <>
    <Navbar/>
    <div className='Overview'>
      <h1>Overview</h1>
      <h3>This is a user interface for various IOT-Devices made of ESP32 microcontrollers equipped with accelerometer and gyroscopes </h3>
    </div>
    <div className='devicelist'>
      <DeviceList set_mac_addr={setMacAddr}/>
    </div>
    <div>
      <p>{macaddr}</p>
    </div>
    <div className="analytics">
      <Analytics macid={macaddr}/>
    </div>
    </>
  );
}

export default App;
