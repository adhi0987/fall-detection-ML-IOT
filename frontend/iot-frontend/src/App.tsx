import './App.css'
import Navbar from './components/Navbar';
import DeviceList from './components/Devicelist';
function App() {

  return (
    <>
    <Navbar/>
    <div className='Overview'>
      <h1>Overview</h1>
      <h3>This is a user interface for various IOT-Devices made of ESP32 microcontrollers equipped with accelerometer and gyroscopes </h3>
    </div>
    <div className='devicelist'>
      <DeviceList/>
    </div>
    </>
  );
}

export default App;
