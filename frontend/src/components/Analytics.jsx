// src/components/Analytics.jsx - FIXED IMPORTS
import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  AlertCircle,
  RefreshCw,
  BarChart3,
  PieChart as PieChartIcon,
  Target,
  Zap,
  Grid, // FIXED: Changed from Grid3X3 to Grid
  ArrowUpDown, // FIXED: Changed from TrendingUpDown to ArrowUpDown
  Volume2,
} from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';
import MultiSymbolComparison from './MultiSymbolComparison';
import VolumeProfileAnalysis from './VolumeProfileAnalysis';
import MarketHeatmap from './MarketHeatmap';
import CandlestickChart from './CandlestickChart';
import apiService from '../services/apiService';

const Analytics = () => {
  const [scanData, setScanData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [selectedChart, setSelectedChart] = useState('overview');
  const [selectedStock, setSelectedStock] = useState(null);

  // Fetch data
  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getLatestScan();

      if (response.success) {
        setScanData(response.data);
        setLastUpdated(new Date());
      } else {
        setError(response.message || 'No scan data available');
      }
    } catch (err) {
      setError(err.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Prepare chart data for overview charts
  const chartData = useMemo(() => {
    if (!scanData) return {};

    // Breakout distribution data
    const distributionData = [
      {
        name: 'Bullish Breakouts',
        value: scanData.breakouts.bullish?.length || 0,
        color: '#22c55e',
      },
      {
        name: 'Bearish Breakouts',
        value: scanData.breakouts.bearish?.length || 0,
        color: '#ef4444',
      },
      {
        name: 'Near Bullish',
        value: scanData.breakouts.near_bullish?.length || 0,
        color: '#3b82f6',
      },
      {
        name: 'Near Bearish',
        value: scanData.breakouts.near_bearish?.length || 0,
        color: '#f59e0b',
      },
    ].filter((item) => item.value > 0);

    // Top performers data
    const allBreakouts = [
      ...(scanData.breakouts.bullish || []),
      ...(scanData.breakouts.bearish || []),
      ...(scanData.breakouts.near_bullish || []),
      ...(scanData.breakouts.near_bearish || []),
    ];

    const topPerformers = allBreakouts
      .sort(
        (a, b) => Math.abs(b.price_change_pct) - Math.abs(a.price_change_pct)
      )
      .slice(0, 10)
      .map((stock) => ({
        symbol: stock.symbol,
        change: stock.price_change_pct,
        volume: stock.volume_ratio,
        type: stock.bullish_breakout
          ? 'Bullish'
          : stock.bearish_breakout
          ? 'Bearish'
          : 'Near',
      }));

    // Volume vs Price Change scatter data
    const scatterData = allBreakouts.map((stock) => ({
      x: stock.volume_ratio,
      y: stock.price_change_pct,
      symbol: stock.symbol,
      type: stock.bullish_breakout
        ? 'Bullish'
        : stock.bearish_breakout
        ? 'Bearish'
        : 'Near',
    }));

    return {
      distribution: distributionData,
      topPerformers,
      scatter: scatterData,
    };
  }, [scanData]);

  // Custom tooltip components
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded shadow-lg">
          <p className="font-medium">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value}
              {entry.name.includes('%') ? '%' : ''}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const ScatterTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded shadow-lg">
          <p className="font-medium">{data.symbol}</p>
          <p>Volume Ratio: {data.x?.toFixed(1)}x</p>
          <p>Price Change: {data.y?.toFixed(1)}%</p>
          <p>Type: {data.type}</p>
        </div>
      );
    }
    return null;
  };

  // Handle stock selection for detailed analysis
  const handleStockSelect = (stock) => {
    setSelectedStock(stock);
    setSelectedChart('stock-detail');
  };

  if (loading) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Advanced Analytics
        </h1>
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Advanced Analytics
        </h1>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
          <p className="text-red-700">{error}</p>
          <button
            onClick={fetchData}
            className="mt-3 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const totalBreakouts =
    (scanData?.breakouts.bullish?.length || 0) +
    (scanData?.breakouts.bearish?.length || 0) +
    (scanData?.breakouts.near_bullish?.length || 0) +
    (scanData?.breakouts.near_bearish?.length || 0);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Advanced Analytics
          </h1>
          <p className="text-gray-600 mt-1">
            Professional-grade analysis of {totalBreakouts} breakout signals
            across {scanData?.metadata.total_scanned || 0} stocks
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-sm text-gray-500">
            Updated: {lastUpdated?.toLocaleTimeString()}
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Chart Navigation */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex flex-wrap gap-2">
          {[
            { id: 'overview', label: 'Market Overview', icon: BarChart3 },
            {
              id: 'performance',
              label: 'Performance Analysis',
              icon: TrendingUp,
            },
            {
              id: 'multi-compare',
              label: 'Multi-Symbol Comparison',
              icon: ArrowUpDown,
            },
            { id: 'volume-profile', label: 'Volume Profile', icon: Volume2 },
            { id: 'heatmap', label: 'Market Heatmap', icon: Grid },
            {
              id: 'advanced-charts',
              label: 'Advanced Charts',
              icon: BarChart3,
            },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setSelectedChart(id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                selectedChart === id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart Content */}
      {selectedChart === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Breakout Distribution Pie Chart */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <PieChartIcon className="h-5 w-5" />
              Breakout Distribution
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={chartData.distribution}
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, value, percent }) =>
                    `${name}: ${value} (${(percent * 100).toFixed(1)}%)`
                  }
                >
                  {chartData.distribution?.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Market Sentiment Gauge */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Target className="h-5 w-5" />
              Market Sentiment
            </h3>
            <div className="flex flex-col items-center justify-center h-[300px]">
              {(() => {
                const bullish = scanData?.breakouts.bullish?.length || 0;
                const bearish = scanData?.breakouts.bearish?.length || 0;
                const total = bullish + bearish;
                const bullishPercent = total > 0 ? (bullish / total) * 100 : 50;

                let sentiment = 'Neutral';
                let color = 'text-gray-600';
                let bgColor = 'bg-gray-100';

                if (bullishPercent > 60) {
                  sentiment = 'Bullish';
                  color = 'text-green-600';
                  bgColor = 'bg-green-100';
                } else if (bullishPercent < 40) {
                  sentiment = 'Bearish';
                  color = 'text-red-600';
                  bgColor = 'bg-red-100';
                }

                return (
                  <div className="text-center">
                    <div
                      className={`w-32 h-32 rounded-full ${bgColor} flex items-center justify-center mb-4`}
                    >
                      <div className={`text-3xl font-bold ${color}`}>
                        {bullishPercent.toFixed(0)}%
                      </div>
                    </div>
                    <div className={`text-xl font-semibold ${color} mb-2`}>
                      {sentiment}
                    </div>
                    <div className="text-sm text-gray-600">
                      {bullish} Bullish vs {bearish} Bearish
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}

      {selectedChart === 'performance' && (
        <div className="space-y-6">
          {/* Top Performers Chart */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Top 10 Price Movers
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={chartData.topPerformers}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="symbol" />
                <YAxis
                  label={{
                    value: 'Price Change (%)',
                    angle: -90,
                    position: 'insideLeft',
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar
                  dataKey="change"
                  fill={(entry) => (entry.change > 0 ? '#22c55e' : '#ef4444')}
                  name="Price Change %"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Volume vs Price Change Scatter */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Volume vs Price Change Analysis
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
              >
                <CartesianGrid />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Volume Ratio"
                  label={{
                    value: 'Volume Ratio',
                    position: 'insideBottom',
                    offset: -10,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Price Change %"
                  label={{
                    value: 'Price Change (%)',
                    angle: -90,
                    position: 'insideLeft',
                  }}
                />
                <Tooltip content={<ScatterTooltip />} />
                <Scatter
                  name="Breakouts"
                  data={chartData.scatter}
                  fill="#3b82f6"
                  fillOpacity={0.6}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {selectedChart === 'multi-compare' && (
        <MultiSymbolComparison scanData={scanData} />
      )}

      {selectedChart === 'volume-profile' && (
        <VolumeProfileAnalysis scanData={scanData} />
      )}

      {selectedChart === 'heatmap' && (
        <MarketHeatmap scanData={scanData} onStockSelect={handleStockSelect} />
      )}

      {selectedChart === 'advanced-charts' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Advanced Technical Analysis
            </h3>
            <p className="text-gray-600 mb-4">
              Select a stock from the heatmap or top performers to view advanced
              candlestick analysis.
            </p>

            {/* Stock Selector */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Stock for Advanced Analysis:
              </label>
              <select
                onChange={(e) => {
                  if (e.target.value) {
                    const allBreakouts = [
                      ...(scanData?.breakouts.bullish || []),
                      ...(scanData?.breakouts.bearish || []),
                      ...(scanData?.breakouts.near_bullish || []),
                      ...(scanData?.breakouts.near_bearish || []),
                    ];
                    const stock = allBreakouts.find(
                      (s) => s.symbol === e.target.value
                    );
                    if (stock) {
                      setSelectedStock(stock);
                    }
                  }
                }}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              >
                <option value="">Choose a stock...</option>
                {chartData.topPerformers?.map((stock) => (
                  <option key={stock.symbol} value={stock.symbol}>
                    {stock.symbol} ({stock.change > 0 ? '+' : ''}
                    {stock.change?.toFixed(1)}%)
                  </option>
                ))}
              </select>
            </div>

            {selectedStock && (
              <CandlestickChart
                symbol={selectedStock.symbol}
                stockData={selectedStock}
              />
            )}
          </div>
        </div>
      )}

      {/* Summary Statistics */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Key Statistics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-blue-600">
              {(
                (totalBreakouts / (scanData?.metadata.total_scanned || 1)) *
                100
              ).toFixed(1)}
              %
            </div>
            <div className="text-sm text-blue-700">Breakout Rate</div>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-green-600">
              {chartData.topPerformers?.[0]?.change?.toFixed(1) || 0}%
            </div>
            <div className="text-sm text-green-700">Top Performer</div>
          </div>
          <div className="bg-yellow-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-yellow-600">
              {(
                chartData.scatter?.reduce((acc, item) => acc + item.x, 0) /
                (chartData.scatter?.length || 1)
              )?.toFixed(1) || 0}
              x
            </div>
            <div className="text-sm text-yellow-700">Avg Volume Ratio</div>
          </div>
          <div className="bg-purple-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-purple-600">
              {scanData?.metadata.period || 20}
            </div>
            <div className="text-sm text-purple-700">Donchian Period</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
