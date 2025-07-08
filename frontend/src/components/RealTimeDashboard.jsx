// src/components/RealTimeDashboard.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  BarChart,
  Bar,
} from 'recharts';
import {
  Activity,
  Wifi,
  WifiOff,
  Play,
  Pause,
  RefreshCw,
  Zap,
  TrendingUp,
  TrendingDown,
  Clock,
  Bell,
  Settings,
} from 'lucide-react';

const RealTimeDashboard = ({ scanData, className = '' }) => {
  const [isLive, setIsLive] = useState(false);
  const [updateInterval, setUpdateInterval] = useState(30); // seconds
  const [liveData, setLiveData] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [lastUpdate, setLastUpdate] = useState(null);

  const intervalRef = useRef(null);
  const wsRef = useRef(null);

  // Initialize live data from scan data
  useEffect(() => {
    if (scanData) {
      initializeLiveData();
    }
  }, [scanData]);

  const initializeLiveData = () => {
    if (!scanData) return;

    const allBreakouts = [
      ...(scanData.breakouts.bullish || []),
      ...(scanData.breakouts.bearish || []),
      ...(scanData.breakouts.near_bullish || []),
      ...(scanData.breakouts.near_bearish || []),
    ];

    // Initialize with current data points
    const initialData = allBreakouts.slice(0, 10).map((stock) => ({
      symbol: stock.symbol,
      price: stock.close_price,
      change: stock.price_change_pct,
      volume: stock.volume_ratio,
      type: stock.bullish_breakout
        ? 'bullish'
        : stock.bearish_breakout
        ? 'bearish'
        : 'near',
      timestamp: new Date().toISOString(),
      history: [
        {
          time: new Date().toISOString(),
          price: stock.close_price,
          volume: stock.volume_ratio,
        },
      ],
    }));

    setLiveData(initialData);
  };

  // Simulate real-time updates
  const simulateUpdate = useCallback(() => {
    setLiveData((prevData) => {
      return prevData.map((stock) => {
        // Simulate price movement (±2% random change)
        const volatility = 0.02;
        const randomChange = (Math.random() - 0.5) * volatility;
        const newPrice = stock.price * (1 + randomChange);
        const priceChange = ((newPrice - stock.price) / stock.price) * 100;

        // Simulate volume change
        const volumeChange = (Math.random() - 0.5) * 0.5; // ±0.25x change
        const newVolume = Math.max(0.5, stock.volume + volumeChange);

        // Add to history (keep last 50 points)
        const newHistory = [
          ...stock.history.slice(-49),
          {
            time: new Date().toISOString(),
            price: newPrice,
            volume: newVolume,
          },
        ];

        // Check for significant changes to trigger notifications
        if (Math.abs(priceChange) > 1) {
          addNotification({
            symbol: stock.symbol,
            type: priceChange > 0 ? 'price_up' : 'price_down',
            message: `${stock.symbol} ${
              priceChange > 0 ? 'jumped' : 'dropped'
            } ${Math.abs(priceChange).toFixed(1)}%`,
            timestamp: new Date().toISOString(),
            severity: Math.abs(priceChange) > 2 ? 'high' : 'medium',
          });
        }

        return {
          ...stock,
          price: newPrice,
          change: stock.change + priceChange,
          volume: newVolume,
          history: newHistory,
          lastUpdate: new Date().toISOString(),
        };
      });
    });

    setLastUpdate(new Date());
  }, []);

  const addNotification = (notification) => {
    setNotifications((prev) => [
      { ...notification, id: Date.now() },
      ...prev.slice(0, 4), // Keep only last 5 notifications
    ]);
  };

  // Start/Stop live updates
  const toggleLiveMode = () => {
    if (isLive) {
      // Stop live mode
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setConnectionStatus('disconnected');
    } else {
      // Start live mode
      setConnectionStatus('connecting');

      // Simulate connection delay
      setTimeout(() => {
        setConnectionStatus('connected');
        intervalRef.current = setInterval(
          simulateUpdate,
          updateInterval * 1000
        );
      }, 1000);
    }

    setIsLive(!isLive);
  };

  // Update interval change
  const handleIntervalChange = (newInterval) => {
    setUpdateInterval(newInterval);

    if (isLive && intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = setInterval(simulateUpdate, newInterval * 1000);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Custom tooltip for live charts
  const LiveTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900 mb-1">
            {new Date(label).toLocaleTimeString()}
          </p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}:{' '}
              {typeof entry.value === 'number'
                ? entry.value.toFixed(2)
                : entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi className="h-4 w-4 text-green-500" />;
      case 'connecting':
        return <RefreshCw className="h-4 w-4 text-yellow-500 animate-spin" />;
      default:
        return <WifiOff className="h-4 w-4 text-red-500" />;
    }
  };

  const getConnectionStatus = () => {
    switch (connectionStatus) {
      case 'connected':
        return { text: 'Live', color: 'text-green-600 bg-green-50' };
      case 'connecting':
        return { text: 'Connecting', color: 'text-yellow-600 bg-yellow-50' };
      default:
        return { text: 'Offline', color: 'text-red-600 bg-red-50' };
    }
  };

  if (!scanData || liveData.length === 0) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
      >
        <div className="text-center py-12">
          <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No Data Available for Live Updates
          </h3>
          <p className="text-gray-600">
            Run your breakout scan to enable real-time monitoring.
          </p>
        </div>
      </div>
    );
  }

  const status = getConnectionStatus();

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <Zap className="h-6 w-6" />
            Real-Time Dashboard
          </h3>
          <p className="text-gray-600 mt-1">
            Live monitoring of {liveData.length} breakout stocks
          </p>
        </div>

        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${status.color}`}
          >
            {getConnectionIcon()}
            {status.text}
          </div>

          {/* Last Update */}
          {lastUpdate && (
            <div className="text-sm text-gray-500">
              Last: {lastUpdate.toLocaleTimeString()}
            </div>
          )}

          {/* Controls */}
          <button
            onClick={toggleLiveMode}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              isLive
                ? 'bg-red-100 text-red-700 hover:bg-red-200'
                : 'bg-green-100 text-green-700 hover:bg-green-200'
            }`}
          >
            {isLive ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            {isLive ? 'Stop Live' : 'Start Live'}
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Settings className="h-4 w-4 text-gray-600" />
              <span className="text-sm font-medium text-gray-700">
                Update Interval:
              </span>
            </div>
            <div className="flex gap-2">
              {[5, 15, 30, 60].map((interval) => (
                <button
                  key={interval}
                  onClick={() => handleIntervalChange(interval)}
                  className={`px-3 py-1 text-xs rounded ${
                    updateInterval === interval
                      ? 'bg-blue-600 text-white'
                      : 'bg-white text-gray-600 border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {interval}s
                </button>
              ))}
            </div>
          </div>

          <div className="text-sm text-gray-600">
            Next update in: {isLive ? updateInterval : 0}s
          </div>
        </div>
      </div>

      {/* Live Notifications */}
      {notifications.length > 0 && (
        <div className="mb-6">
          <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
            <Bell className="h-4 w-4" />
            Live Alerts
          </h4>
          <div className="space-y-2">
            {notifications.slice(0, 3).map((notification) => (
              <div
                key={notification.id}
                className={`flex items-center gap-3 p-3 rounded-lg border-l-4 ${
                  notification.severity === 'high'
                    ? 'bg-red-50 border-red-400'
                    : 'bg-yellow-50 border-yellow-400'
                }`}
              >
                <div
                  className={`p-1 rounded-full ${
                    notification.type === 'price_up'
                      ? 'bg-green-100'
                      : 'bg-red-100'
                  }`}
                >
                  {notification.type === 'price_up' ? (
                    <TrendingUp className="h-3 w-3 text-green-600" />
                  ) : (
                    <TrendingDown className="h-3 w-3 text-red-600" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="text-sm font-medium text-gray-900">
                    {notification.message}
                  </div>
                  <div className="text-xs text-gray-500">
                    {new Date(notification.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Live Price Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Real-time Price Movement */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-3">
            Live Price Movement
          </h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={liveData[0]?.history || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="time"
                  tickFormatter={(time) =>
                    new Date(time).toLocaleTimeString().slice(0, 5)
                  }
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  tick={{ fontSize: 10 }}
                  domain={['dataMin - 0.1', 'dataMax + 0.1']}
                />
                <Tooltip content={<LiveTooltip />} />
                <Line
                  type="monotone"
                  dataKey="price"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  name="Price"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="text-xs text-gray-600 mt-2">
            {liveData[0]?.symbol || 'No data'} - Last 50 updates
          </div>
        </div>

        {/* Volume Activity */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-3">Volume Activity</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={liveData.slice(0, 8)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="symbol" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="volume" name="Volume Ratio">
                  {liveData.slice(0, 8).map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={
                        entry.volume > 2
                          ? '#22c55e'
                          : entry.volume > 1.5
                          ? '#f59e0b'
                          : '#94a3b8'
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Live Stock Table */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-4">Live Stock Monitor</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 font-medium text-gray-600">
                  Symbol
                </th>
                <th className="text-right py-2 font-medium text-gray-600">
                  Price
                </th>
                <th className="text-right py-2 font-medium text-gray-600">
                  Change %
                </th>
                <th className="text-right py-2 font-medium text-gray-600">
                  Volume
                </th>
                <th className="text-center py-2 font-medium text-gray-600">
                  Type
                </th>
                <th className="text-center py-2 font-medium text-gray-600">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {liveData.map((stock, index) => (
                <tr
                  key={stock.symbol}
                  className={`border-b border-gray-100 ${
                    isLive ? 'animate-pulse' : ''
                  }`}
                >
                  <td className="py-3 font-medium text-gray-900">
                    {stock.symbol}
                  </td>
                  <td className="text-right py-3">${stock.price.toFixed(2)}</td>
                  <td
                    className={`text-right py-3 font-medium ${
                      stock.change > 0 ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {stock.change > 0 ? '+' : ''}
                    {stock.change.toFixed(1)}%
                  </td>
                  <td className="text-right py-3">
                    {stock.volume.toFixed(1)}x
                  </td>
                  <td className="text-center py-3">
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${
                        stock.type === 'bullish'
                          ? 'bg-green-100 text-green-700'
                          : stock.type === 'bearish'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-blue-100 text-blue-700'
                      }`}
                    >
                      {stock.type}
                    </span>
                  </td>
                  <td className="text-center py-3">
                    <div
                      className={`w-2 h-2 rounded-full mx-auto ${
                        isLive ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                      }`}
                    ></div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">
            {liveData.filter((s) => s.change > 0).length}
          </div>
          <div className="text-sm text-blue-700">Gaining</div>
        </div>
        <div className="bg-red-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-red-600">
            {liveData.filter((s) => s.change < 0).length}
          </div>
          <div className="text-sm text-red-700">Declining</div>
        </div>
        <div className="bg-green-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-600">
            {liveData.filter((s) => s.volume > 2).length}
          </div>
          <div className="text-sm text-green-700">High Volume</div>
        </div>
        <div className="bg-purple-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-purple-600">
            {notifications.length}
          </div>
          <div className="text-sm text-purple-700">Alerts</div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="mt-6 text-xs text-gray-500 text-center">
        Real-time simulation for demonstration purposes. In production, this
        would connect to live market data feeds.
      </div>
    </div>
  );
};

export default RealTimeDashboard;
