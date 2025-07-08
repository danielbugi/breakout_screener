import React, { useState } from 'react';
import { CheckCircle, XCircle, Clock, Play, Wifi } from 'lucide-react';
import apiService from '../services/apiService';

const ApiTest = () => {
  const [testing, setTesting] = useState(false);
  const [results, setResults] = useState(null);

  const runTests = async () => {
    console.log('🧪 API Test: Manual test started');
    setTesting(true);
    setResults(null);

    const testResults = {
      health: { status: 'pending', message: 'Testing...' },
      latestScan: { status: 'pending', message: 'Testing...' },
      summary: { status: 'pending', message: 'Testing...' },
      connection: { status: 'pending', message: 'Testing...' },
    };

    setResults({ ...testResults });

    // Test 1: Health Check
    try {
      console.log('🧪 Testing health check...');
      const healthResult = await apiService.healthCheck();
      testResults.health = healthResult.connected
        ? { status: 'success', message: 'Backend is healthy and responsive' }
        : {
            status: 'error',
            message: healthResult.error || 'Backend not responding',
          };
    } catch (error) {
      testResults.health = { status: 'error', message: error.message };
    }
    setResults({ ...testResults });

    // Test 2: Connection Speed
    try {
      console.log('🧪 Testing connection speed...');
      const connectionResult = await apiService.testConnection();
      testResults.connection = connectionResult.success
        ? {
            status: 'success',
            message: `Connected in ${connectionResult.duration}ms`,
          }
        : { status: 'error', message: connectionResult.error };
    } catch (error) {
      testResults.connection = { status: 'error', message: error.message };
    }
    setResults({ ...testResults });

    // Test 3: Latest Scan Data
    try {
      console.log('🧪 Testing latest scan endpoint...');
      const scanResult = await apiService.getLatestScan();
      testResults.latestScan = scanResult.success
        ? {
            status: 'success',
            message: `Scan data available (${
              scanResult.data?.metadata?.total_scanned || 0
            } stocks)`,
          }
        : {
            status: 'warning',
            message: scanResult.message || 'No scan data available',
          };
    } catch (error) {
      testResults.latestScan = { status: 'error', message: error.message };
    }
    setResults({ ...testResults });

    // Test 4: Summary Data
    try {
      console.log('🧪 Testing summary endpoint...');
      const summaryResult = await apiService.getScanSummary();
      testResults.summary = summaryResult.success
        ? { status: 'success', message: 'Summary data loaded successfully' }
        : {
            status: 'warning',
            message: summaryResult.message || 'No summary data available',
          };
    } catch (error) {
      testResults.summary = { status: 'error', message: error.message };
    }
    setResults({ ...testResults });

    setTesting(false);
    console.log('🧪 API Test: Complete');
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-red-600" />;
      case 'warning':
        return <Clock className="h-5 w-5 text-yellow-600" />;
      case 'pending':
        return (
          <div className="h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        );
      default:
        return <Clock className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return 'bg-green-50 border-green-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200';
      case 'pending':
        return 'bg-blue-50 border-blue-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  console.log('🧪 API Test component rendered - NO AUTO-TESTS');

  return (
    <div className="p-6">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-6">
          <div className="flex items-center justify-center mb-4">
            <Wifi className="h-8 w-8 text-blue-600 mr-2" />
            <h1 className="text-2xl font-bold text-gray-900">
              API Connection Test
            </h1>
          </div>
          <p className="text-gray-600">
            Test the connection between frontend and backend services
          </p>
          <div className="mt-2 text-sm text-blue-600 bg-blue-50 px-3 py-1 rounded">
            Manual testing only - No automatic tests
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Connection Tests
            </h2>
            <button
              onClick={runTests}
              disabled={testing}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              <Play className={`h-4 w-4 ${testing ? 'animate-pulse' : ''}`} />
              {testing ? 'Testing...' : 'Run Tests'}
            </button>
          </div>

          <div className="space-y-3">
            {results ? (
              Object.entries(results).map(([test, result]) => (
                <div
                  key={test}
                  className={`p-3 rounded-lg border ${getStatusColor(
                    result.status
                  )}`}
                >
                  <div className="flex items-center gap-3">
                    {getStatusIcon(result.status)}
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 capitalize">
                        {test.replace(/([A-Z])/g, ' $1').trim()}
                      </div>
                      <div className="text-sm text-gray-600">
                        {result.message}
                      </div>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                Click "Run Tests" to check API connectivity
                <br />
                <span className="text-sm text-blue-600">
                  Tests run only when you click the button
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="font-medium text-gray-900 mb-2">Prerequisites:</h3>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• Flask backend running on http://localhost:5000</li>
            <li>• Vite dev server running on http://localhost:3000</li>
            <li>• Automation scripts have generated scan data</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ApiTest;
