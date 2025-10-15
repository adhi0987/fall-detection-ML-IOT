import { Route, Routes } from 'react-router-dom';
import './App.css'; // We will add styles later
import Navbar from './components/Navbar';
import PredictionAnalytics from './pages/PredictionAnalytics'; // We will create this next
import Training from './pages/Training'; // We will create this next

function App() {
  return (
    <div className="app-container">
      <Navbar /> {/* The navbar is always visible at the top */}
      <main className="main-content">
        {/* The Routes component decides which page to show based on the URL */}
        <Routes>
          {/* If the URL is '/', show the PredictionAnalytics page */}
          <Route path="/" element={<PredictionAnalytics />} />
          {/* If the URL is '/training', show the Training page */}
          <Route path="/training" element={<Training />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;