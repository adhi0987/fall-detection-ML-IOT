import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom'; // Import BrowserRouter
import App from './App';
import './App.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* This component enables page routing for our entire app */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);