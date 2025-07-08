import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ComposedChart,
  Bar,
  ScatterChart,
  Scatter,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Plus,
  X,
  BarChart3,
  Settings,
  Download,
  RefreshCw,
  Zap,
} from 'lucide-react';

const MultiSymbolComparison = ({ scanData, className = '' }) => {
  const [selectedSymbols, setSelectedSymbols] = useState([]);
  const [chartType, setChartType] = useState('price');
  const [timeframe, setTimeframe] = useState('30d');
  const [showVolume, setShowVolume] = useState(true);
  const [normalizePrices, setNormalizePrices] = useState(true);

  // Available symbols from scan data
  const availableSymbols = useMemo(() => {
    if (!scanData) return [];

    const symbols = [];
    Object.values(scanData.breakouts || {}).forEach((breakoutArray) => {
      if (Array.isArray(breakoutArray)) {
        breakoutArray.forEach((stock) => {
          if (!symbols.find((s) => s.symbol === stock.symbol)) {
            symbols.push({
              symbol: stock.symbol,
              type: stock.bullish_breakout
                ? 'bullish'
                : stock.bearish_breakout
                ? 'bearish'
                : 'near',
              change: stock.price_change_pct,
              volume: stock.volume_ratio,
              price: stock.close_price,
            });
          }
        });
      }
    });

    return symbols.sort((a, b) => a.symbol.localeCompare(b.symbol));
  }, [scanData]);

  // Initialize with top performers
  useEffect(() => {
    if (availableSymbols.length > 0 && selectedSymbols.length === 0) {
      const topPerformers = availableSymbols
        .sort((a, b) => Math.abs(b.change) - Math.abs(a.change))
        .slice(0, 4)
        .map((s) => s.symbol);
      setSelectedSymbols(topPerformers);
    }
  }, [availableSymbols, selectedSymbols.length]);

  // Generate comparison data
  const comparisonData = useMemo(() => {
    if (selectedSymbols.length === 0) return [];

    const days = timeframe === '30d' ? 30 : timeframe === '60d' ? 60 : 90;
    const data = [];

    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];

      const dayData = { date: dateStr };

      selectedSymbols.forEach((symbol) => {
        const stockInfo = availableSymbols.find((s) => s.symbol === symbol);
        if (!stockInfo) return;

        // Generate mock historical data
        const volatility = 0.02;
        const trend = stockInfo.change > 0 ? 0.001 : -0.001;
        const randomChange = (Math.random() - 0.5) * volatility;

        let price;
        if (i === 0) {
          price = stockInfo.price;
        } else {
          const daysFromEnd = i;
          const trendEffect = trend * daysFromEnd;
          const basePrice = stockInfo.price / (1 + stockInfo.change / 100);
          price = basePrice * (1 + trendEffect + randomChange);
        }

        // Normalize prices if enabled
        const normalizedPrice = normalizePrices
          ? (price / stockInfo.price) * 100
          : price;

        dayData[`${symbol}_price`] = normalizedPrice;
        dayData[`${symbol}_volume`] = Math.random() * 2000000 + 500000;
        dayData[`${symbol}_rsi`] = 30 + Math.random() * 40;
        dayData[`${symbol}_change`] =
          i === 0 ? stockInfo.change : (Math.random() - 0.5) * 5;
      });

      data.push(dayData);
    }

    return data;
  }, [selectedSymbols, availableSymbols, timeframe, normalizePrices]);

  // Color palette for different symbols
  const colorPalette = [
    '#3b82f6',
    '#ef4444',
    '#22c55e',
    '#f59e0b',
    '#8b5cf6',
    '#ec4899',
    '#06b6d4',
    '#84cc16',
  ];

  const getSymbolColor = (index) => colorPalette[index % colorPalette.length];

  const addSymbol = (symbol) => {
    if (!selectedSymbols.includes(symbol) && selectedSymbols.length < 8) {
      setSelectedSymbols([...selectedSymbols, symbol]);
    }
  };

  const removeSymbol = (symbol) => {
    setSelectedSymbols(selectedSymbols.filter((s) => s !== symbol));
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg min-w-64">
          <p className="font-medium text-gray-900 mb-2">{label}</p>
          <div className="space-y-1">
            {payload.map((entry, index) => {
              const symbol = entry.dataKey.split('_')[0];
              const metric = entry.dataKey.split('_')[1];

              return (
                <div key={index} className="flex justify-between items-center">
                  <span style={{ color: entry.color }} className="font-medium">
                    {symbol}:
                  </span>
                  <span className="ml-2">
                    {metric === 'price'
                      ? normalizePrices
                        ? `${entry.value?.toFixed(1)}%`
                        : `$${entry.value?.toFixed(2)}`
                      : metric === 'volume'
                      ? `${(entry.value / 1000000).toFixed(1)}M`
                      : `${entry.value?.toFixed(1)}%`}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }
    return null;
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  const exportData = () => {
    if (comparisonData.length === 0) return;

    const headers = [
      'Date',
      ...selectedSymbols.map((s) => `${s}_Price`),
      ...selectedSymbols.map((s) => `${s}_Volume`),
    ];

    const csvContent = [
      headers.join(','),
      ...comparisonData.map((row) =>
        [
          row.date,
          ...selectedSymbols.map((s) => row[`${s}_price`]?.toFixed(2) || ''),
          ...selectedSymbols.map((s) => row[`${s}_volume`]?.toFixed(0) || ''),
        ].join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `comparison_${selectedSymbols.join('_')}_${
      new Date().toISOString().split('T')[0]
    }.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (!scanData || availableSymbols.length === 0) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
      >
        <div className="text-center py-12">
          <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No Data Available for Comparison
          </h3>
          <p className="text-gray-600">
            Run your breakout scan to generate comparison data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <BarChart3 className="h-6 w-6" />
            Multi-Symbol Comparison
          </h3>
          <p className="text-gray-600 mt-1">
            Compare performance of {selectedSymbols.length} selected stocks
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={exportData}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            <Download className="h-4 w-4" />
            Export
          </button>
          <button
            onClick={() => window.location.reload()}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
        {/* Chart Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Chart Type
          </label>
          <select
            value={chartType}
            onChange={(e) => setChartType(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          >
            <option value="price">Price Comparison</option>
            <option value="volume">Volume Analysis</option>
            <option value="performance">Performance Metrics</option>
            <option value="correlation">Correlation Analysis</option>
          </select>
        </div>

        {/* Timeframe */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Timeframe
          </label>
          <div className="flex gap-1 bg-white rounded-lg p-1 border border-gray-300">
            {['30d', '60d', '90d'].map((period) => (
              <button
                key={period}
                onClick={() => setTimeframe(period)}
                className={`flex-1 px-2 py-1 text-xs rounded ${
                  timeframe === period
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {period}
              </button>
            ))}
          </div>
        </div>

        {/* Options */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Options
          </label>
          <div className="space-y-2">
            <label className="flex items-center text-sm">
              <input
                type="checkbox"
                checked={normalizePrices}
                onChange={(e) => setNormalizePrices(e.target.checked)}
                className="mr-2"
              />
              Normalize Prices
            </label>
            <label className="flex items-center text-sm">
              <input
                type="checkbox"
                checked={showVolume}
                onChange={(e) => setShowVolume(e.target.checked)}
                className="mr-2"
              />
              Show Volume
            </label>
          </div>
        </div>

        {/* Add Symbol */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Add Symbol
          </label>
          <select
            onChange={(e) => {
              if (e.target.value) {
                addSymbol(e.target.value);
                e.target.value = '';
              }
            }}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">Select symbol...</option>
            {availableSymbols
              .filter((s) => !selectedSymbols.includes(s.symbol))
              .map((stock) => (
                <option key={stock.symbol} value={stock.symbol}>
                  {stock.symbol} ({stock.change > 0 ? '+' : ''}
                  {stock.change?.toFixed(1)}%)
                </option>
              ))}
          </select>
        </div>
      </div>

      {/* Selected Symbols */}
      <div className="mb-6">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-medium text-gray-700">Selected:</span>
          {selectedSymbols.map((symbol, index) => {
            const stockInfo = availableSymbols.find((s) => s.symbol === symbol);
            return (
              <div
                key={symbol}
                className="flex items-center gap-2 px-3 py-1 bg-white border-2 rounded-lg text-sm"
                style={{ borderColor: getSymbolColor(index) }}
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getSymbolColor(index) }}
                ></div>
                <span className="font-medium">{symbol}</span>
                {stockInfo && (
                  <span
                    className={`text-xs ${
                      stockInfo.change > 0 ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {stockInfo.change > 0 ? '+' : ''}
                    {stockInfo.change?.toFixed(1)}%
                  </span>
                )}
                <button
                  onClick={() => removeSymbol(symbol)}
                  className="text-gray-400 hover:text-red-500"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Chart */}
      <div className="h-96 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'price' ? (
            <LineChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fontSize: 12 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                tickFormatter={(value) =>
                  normalizePrices
                    ? `${value.toFixed(0)}%`
                    : `$${value.toFixed(0)}`
                }
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {selectedSymbols.map((symbol, index) => (
                <Line
                  key={symbol}
                  type="monotone"
                  dataKey={`${symbol}_price`}
                  stroke={getSymbolColor(index)}
                  strokeWidth={2}
                  dot={false}
                  name={symbol}
                />
              ))}
            </LineChart>
          ) : chartType === 'volume' ? (
            <ComposedChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fontSize: 12 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {selectedSymbols.map((symbol, index) => (
                <Bar
                  key={symbol}
                  dataKey={`${symbol}_volume`}
                  fill={getSymbolColor(index)}
                  fillOpacity={0.7}
                  name={`${symbol} Volume`}
                />
              ))}
            </ComposedChart>
          ) : chartType === 'performance' ? (
            <ScatterChart data={comparisonData.slice(-1)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                type="number"
                dataKey={`${selectedSymbols[0]}_change`}
                name="Price Change %"
                tick={{ fontSize: 12 }}
              />
              <YAxis
                type="number"
                dataKey={`${selectedSymbols[0]}_volume`}
                name="Volume"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
              />
              <Tooltip content={<CustomTooltip />} />
              {selectedSymbols.map((symbol, index) => {
                const stockInfo = availableSymbols.find(
                  (s) => s.symbol === symbol
                );
                return (
                  <Scatter
                    key={symbol}
                    name={symbol}
                    data={[
                      {
                        x: stockInfo?.change || 0,
                        y: stockInfo?.volume || 1,
                        symbol: symbol,
                      },
                    ]}
                    fill={getSymbolColor(index)}
                  />
                );
              })}
            </ScatterChart>
          ) : (
            <LineChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fontSize: 12 }}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {selectedSymbols.map((symbol, index) => (
                <Line
                  key={symbol}
                  type="monotone"
                  dataKey={`${symbol}_rsi`}
                  stroke={getSymbolColor(index)}
                  strokeWidth={2}
                  dot={false}
                  name={`${symbol} RSI`}
                />
              ))}
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Comparison Table */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-3">Current Metrics</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-gray-200">
                <th className="pb-2">Symbol</th>
                <th className="pb-2">Price</th>
                <th className="pb-2">Change %</th>
                <th className="pb-2">Volume Ratio</th>
                <th className="pb-2">Type</th>
              </tr>
            </thead>
            <tbody>
              {selectedSymbols.map((symbol, index) => {
                const stockInfo = availableSymbols.find(
                  (s) => s.symbol === symbol
                );
                if (!stockInfo) return null;

                return (
                  <tr key={symbol} className="border-b border-gray-100">
                    <td className="py-2 flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getSymbolColor(index) }}
                      ></div>
                      <span className="font-medium">{symbol}</span>
                    </td>
                    <td className="py-2">${stockInfo.price?.toFixed(2)}</td>
                    <td
                      className={`py-2 font-medium ${
                        stockInfo.change > 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {stockInfo.change > 0 ? '+' : ''}
                      {stockInfo.change?.toFixed(1)}%
                    </td>
                    <td className="py-2">{stockInfo.volume?.toFixed(1)}x</td>
                    <td className="py-2">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          stockInfo.type === 'bullish'
                            ? 'bg-green-100 text-green-700'
                            : stockInfo.type === 'bearish'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}
                      >
                        {stockInfo.type}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default MultiSymbolComparison;
