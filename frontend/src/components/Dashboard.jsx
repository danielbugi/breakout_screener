import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  AlertCircle,
  BarChart3,
  RefreshCw,
  Calendar,
  Wifi,
  WifiOff,
} from 'lucide-react';
import StatCard from './StatCard';
import BreakoutTable from './BreakoutTable';
import LoadingSpinner from './LoadingSpinner';
import apiService from '../services/apiService';

const Dashboard = () => {
  const [scanData, setScanData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('checking');

  // Use refs to prevent multiple simultaneous requests
  const fetchingRef = useRef(false);
  const intervalRef = useRef(null);

  // Fetch data function with proper error handling
  const fetchData = useCallback(async (isManualRefresh = false) => {
    // Prevent multiple simultaneous requests
    if (fetchingRef.current && !isManualRefresh) {
      return;
    }

    fetchingRef.current = true;
    if (isManualRefresh) setLoading(true);
    setError(null);
    setConnectionStatus('connecting');

    try {
      // Check backend health
      const healthResult = await apiService.healthCheck();

      if (!healthResult.connected) {
        throw new Error('Backend not responding');
      }

      setConnectionStatus('connected');

      // Fetch scan data
      const scanResponse = await apiService.getLatestScan();

      if (scanResponse.success) {
        setScanData(scanResponse.data);
        setLastUpdated(new Date());
      } else {
        // Not an error - just no data available
        setError(
          `No scan data available: ${
            scanResponse.message || 'Run automation to generate data'
          }`
        );
      }
    } catch (err) {
      setConnectionStatus('disconnected');
      setError(err.message || 'Failed to connect to backend');
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
      fetchingRef.current = false;
    }
  }, []);

  // Initialize data fetch on component mount
  useEffect(() => {
    fetchData();

    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      fetchingRef.current = false;
    };
  }, [fetchData]);

  // Set up auto-refresh interval (every 2 minutes when connected)
  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Only set up auto-refresh if connected and has data
    if (connectionStatus === 'connected' && scanData) {
      intervalRef.current = setInterval(() => {
        fetchData(); // Auto-refresh without loading indicator
      }, 120000); // 2 minutes
    }

    // Cleanup interval on dependency change
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [connectionStatus, scanData, fetchData]);

  const handleManualRefresh = useCallback(() => {
    fetchData(true); // Manual refresh with loading indicator
  }, [fetchData]);

  const handleViewMore = useCallback((type) => {
    // Navigate to breakouts page with filter
    console.log(`Navigate to breakouts page with type: ${type}`);
    // In a real app: navigate(`/breakouts?type=${type}`);
  }, []);

  const formatTime = (date) => {
    if (!date) return 'Never';
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      month: 'short',
      day: 'numeric',
    }).format(date);
  };

  // Show loading state on initial load
  if (loading && !scanData) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Market Overview
            </h1>
            <p className="text-gray-600 mt-1">
              Loading latest breakout analysis...
            </p>
          </div>
        </div>
        <LoadingSpinner />
      </div>
    );
  }

  // Use real data or provide empty structure
  const displayData = scanData || {
    metadata: {
      scan_date: new Date().toISOString(),
      period: 20,
      total_scanned: 0,
      bullish_count: 0,
      bearish_count: 0,
      near_breakout_count: 0,
      failed_count: 0,
    },
    breakouts: {
      bullish: [],
      bearish: [],
      near_bullish: [],
      near_bearish: [],
    },
  };

  const hasData = scanData && displayData.metadata.total_scanned > 0;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Market Overview</h1>
          <p className="text-gray-600 mt-1">
            Real-time Donchian breakout analysis across major indices
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div
            className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
              connectionStatus === 'connected'
                ? 'bg-green-50'
                : connectionStatus === 'connecting'
                ? 'bg-yellow-50'
                : 'bg-red-50'
            }`}
          >
            {connectionStatus === 'connected' ? (
              <>
                <Wifi className="w-2 h-2 text-green-500" />
                <span className="text-sm text-green-700 font-medium">Live</span>
              </>
            ) : connectionStatus === 'connecting' ? (
              <>
                <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-yellow-700 font-medium">
                  Connecting
                </span>
              </>
            ) : (
              <>
                <WifiOff className="w-2 h-2 text-red-500" />
                <span className="text-sm text-red-700 font-medium">
                  Offline
                </span>
              </>
            )}
          </div>

          <div className="text-sm text-gray-500">
            Updated: {formatTime(lastUpdated)}
          </div>

          <button
            onClick={handleManualRefresh}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error/No Data Banner */}
      {error && (
        <div
          className={`rounded-lg p-4 ${
            hasData
              ? 'bg-yellow-50 border border-yellow-200'
              : 'bg-blue-50 border border-blue-200'
          }`}
        >
          <div className="flex items-center">
            <AlertCircle
              className={`h-5 w-5 mr-2 ${
                hasData ? 'text-yellow-600' : 'text-blue-600'
              }`}
            />
            <span className={hasData ? 'text-yellow-800' : 'text-blue-800'}>
              {hasData ? `Warning: ${error}` : error}
            </span>
            {!hasData && (
              <span className="ml-2 text-blue-700">
                Run your automation scripts to generate scan data.
              </span>
            )}
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Scanned"
          value={displayData.metadata.total_scanned.toLocaleString()}
          subtitle={hasData ? 'Stocks analyzed' : 'No scan data'}
          icon={Activity}
          color="text-blue-600"
          bgColor="bg-blue-50"
        />
        <StatCard
          title="Bullish Breakouts"
          value={displayData.metadata.bullish_count}
          subtitle={
            hasData
              ? `${(
                  (displayData.metadata.bullish_count /
                    displayData.metadata.total_scanned) *
                  100
                ).toFixed(1)}% of total`
              : 'No data'
          }
          icon={TrendingUp}
          color="text-green-600"
          bgColor="bg-green-50"
        />
        <StatCard
          title="Bearish Breakouts"
          value={displayData.metadata.bearish_count}
          subtitle={
            hasData
              ? `${(
                  (displayData.metadata.bearish_count /
                    displayData.metadata.total_scanned) *
                  100
                ).toFixed(1)}% of total`
              : 'No data'
          }
          icon={TrendingDown}
          color="text-red-600"
          bgColor="bg-red-50"
        />
        <StatCard
          title="Near Breakouts"
          value={displayData.metadata.near_breakout_count}
          subtitle={hasData ? 'Stocks to watch' : 'No data'}
          icon={AlertCircle}
          color="text-yellow-600"
          bgColor="bg-yellow-50"
        />
      </div>

      {/* Breakout Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <BreakoutTable
          title="Bullish Breakouts"
          data={displayData.breakouts.bullish}
          type="bullish"
          icon={TrendingUp}
          onViewMore={handleViewMore}
        />
        <BreakoutTable
          title="Bearish Breakouts"
          data={displayData.breakouts.bearish}
          type="bearish"
          icon={TrendingDown}
          onViewMore={handleViewMore}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <BreakoutTable
          title="Near Bullish Breakouts"
          data={displayData.breakouts.near_bullish}
          type="near"
          icon={AlertCircle}
          onViewMore={handleViewMore}
        />
        <BreakoutTable
          title="Near Bearish Breakouts"
          data={displayData.breakouts.near_bearish}
          type="near"
          icon={AlertCircle}
          onViewMore={handleViewMore}
        />
      </div>

      {/* Footer */}
      <div className="text-center text-sm text-gray-500 mt-8 py-4 border-t border-gray-200">
        <div className="flex items-center justify-center gap-4">
          <span>Using {displayData.metadata.period}-day Donchian channels</span>
          <span>•</span>
          <span>
            Scan date:{' '}
            {new Date(displayData.metadata.scan_date).toLocaleDateString()}
          </span>
          <span>•</span>
          <span className="flex items-center gap-1">
            <Calendar className="h-4 w-4" />
            Auto-refresh: Every 2 minutes
          </span>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
