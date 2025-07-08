import React from 'react';

const ScanHistory = () => (
  <div className="p-6">
    <h1 className="text-3xl font-bold text-gray-900 mb-6">Scan History</h1>
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <p className="text-gray-600">
        Historical scan data will be displayed here.
      </p>
      <p className="text-gray-500 mt-4">This page will show:</p>
      <ul className="list-disc list-inside text-gray-500 mt-2 space-y-1">
        <li>Previous scan results</li>
        <li>Historical performance</li>
        <li>Trend analysis over time</li>
        <li>Download historical data</li>
      </ul>
    </div>
  </div>
);

export default ScanHistory;
