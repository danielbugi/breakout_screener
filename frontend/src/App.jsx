import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import BreakoutDetails from './components/BreakoutDetails';
import ScanHistory from './components/ScanHistory';
import Analytics from './components/Analytics';
import ApiTest from './components/ApiTest';

const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="max-w-7xl mx-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/breakouts" element={<BreakoutDetails />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/history" element={<ScanHistory />} />
            <Route path="/api-test" element={<ApiTest />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;