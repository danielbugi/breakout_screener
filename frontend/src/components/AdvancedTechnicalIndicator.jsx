// src/components/AdvancedTechnicalIndicators.jsx
import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Area,
  AreaChart,
  ReferenceLine,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Target,
  BarChart3,
  Settings,
  Eye,
  EyeOff,
  RefreshCw,
} from 'lucide-react';

const AdvancedTechnicalIndicators = ({ scanData, className = '' }) => {
  const [selectedIndicators, setSelectedIndicators] = useState({
    rsi: true,
    macd: true,
    bollinger: false,
    stochastic: false,
    williams: false,
    momentum: true,
  });

  const [timeframe, setTimeframe] = useState('30d');
  const [selectedStock, setSelectedStock] = useState(null);

  // Available stocks from scan data
  const availableStocks = useMemo(() => {
    if (!scanData) return [];

    const stocks = [];
    Object.values(scanData.breakouts || {}).forEach((breakoutArray) => {
      if (Array.isArray(breakoutArray)) {
        breakoutArray.forEach((stock) => {
          stocks.push({
            ...stock,
            type: stock.bullish_breakout
              ? 'bullish'
              : stock.bearish_breakout
              ? 'bearish'
              : 'near',
          });
        });
      }
    });

    return stocks.sort(
      (a, b) => Math.abs(b.price_change_pct) - Math.abs(a.price_change_pct)
    );
  }, [scanData]);

  // Generate technical indicator data
  const technicalData = useMemo(() => {
    if (!selectedStock) return [];

    const days = timeframe === '30d' ? 30 : timeframe === '60d' ? 60 : 90;
    const data = [];
    const currentPrice = selectedStock.close_price;

    // Generate historical price data
    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);

      // Generate realistic price movement
      const volatility = 0.02;
      const trend = selectedStock.price_change_pct > 0 ? 0.001 : -0.001;
      const randomChange = (Math.random() - 0.5) * volatility;

      let price;
      if (i === 0) {
        price = currentPrice;
      } else {
        const basePrice =
          currentPrice / (1 + selectedStock.price_change_pct / 100);
        const trendEffect = trend * i;
        price = basePrice * (1 + trendEffect + randomChange);
      }

      // Generate OHLC data
      const open = price + (Math.random() - 0.5) * price * 0.01;
      const high = Math.max(open, price) + Math.random() * price * 0.005;
      const low = Math.min(open, price) - Math.random() * price * 0.005;
      const volume = Math.random() * 2000000 + 500000;

      data.push({
        date: date.toISOString().split('T')[0],
        open,
        high,
        low,
        close: price,
        volume,
      });
    }

    // Calculate technical indicators
    return calculateIndicators(data);
  }, [selectedStock, timeframe]);

  // Calculate various technical indicators
  const calculateIndicators = (priceData) => {
    const data = [...priceData];

    data.forEach((item, index) => {
      // RSI Calculation (14-period)
      if (index >= 14) {
        const period = 14;
        const changes = data
          .slice(index - period + 1, index + 1)
          .map((d, i, arr) => (i === 0 ? 0 : d.close - arr[i - 1].close));

        const gains =
          changes.filter((c) => c > 0).reduce((sum, c) => sum + c, 0) / period;
        const losses =
          Math.abs(
            changes.filter((c) => c < 0).reduce((sum, c) => sum + c, 0)
          ) / period;

        const rs = gains / (losses || 1);
        item.rsi = 100 - 100 / (1 + rs);
      }

      // MACD Calculation (12, 26, 9)
      if (index >= 26) {
        const ema12 = calculateEMA(data.slice(0, index + 1), 12, 'close');
        const ema26 = calculateEMA(data.slice(0, index + 1), 26, 'close');
        item.macd = ema12 - ema26;

        // Signal line (9-period EMA of MACD)
        if (index >= 34) {
          const macdData = data
            .slice(0, index + 1)
            .filter((d) => d.macd !== undefined);
          item.macdSignal = calculateEMA(macdData, 9, 'macd');
          item.macdHistogram = item.macd - item.macdSignal;
        }
      }

      // Bollinger Bands (20-period, 2 std dev)
      if (index >= 20) {
        const period = 20;
        const prices = data
          .slice(index - period + 1, index + 1)
          .map((d) => d.close);
        const sma = prices.reduce((sum, p) => sum + p, 0) / period;
        const variance =
          prices.reduce((sum, p) => sum + Math.pow(p - sma, 2), 0) / period;
        const stdDev = Math.sqrt(variance);

        item.bollingerMiddle = sma;
        item.bollingerUpper = sma + 2 * stdDev;
        item.bollingerLower = sma - 2 * stdDev;
        item.bollingerWidth =
          ((item.bollingerUpper - item.bollingerLower) / item.bollingerMiddle) *
          100;
      }

      // Stochastic Oscillator (14, 3, 3)
      if (index >= 14) {
        const period = 14;
        const highLow = data.slice(index - period + 1, index + 1);
        const highest = Math.max(...highLow.map((d) => d.high));
        const lowest = Math.min(...highLow.map((d) => d.low));

        const k = ((item.close - lowest) / (highest - lowest)) * 100;
        item.stochK = k;

        // %D is 3-period SMA of %K
        if (index >= 16) {
          const kValues = data
            .slice(index - 2, index + 1)
            .map((d) => d.stochK)
            .filter((v) => v !== undefined);
          item.stochD = kValues.reduce((sum, v) => sum + v, 0) / kValues.length;
        }
      }

      // Williams %R (14-period)
      if (index >= 14) {
        const period = 14;
        const highLow = data.slice(index - period + 1, index + 1);
        const highest = Math.max(...highLow.map((d) => d.high));
        const lowest = Math.min(...highLow.map((d) => d.low));

        item.williamsR = ((highest - item.close) / (highest - lowest)) * -100;
      }

      // Momentum (10-period)
      if (index >= 10) {
        const prevPrice = data[index - 10].close;
        item.momentum = ((item.close - prevPrice) / prevPrice) * 100;
      }

      // Moving Averages
      if (index >= 9) {
        item.sma10 =
          data
            .slice(index - 9, index + 1)
            .reduce((sum, d) => sum + d.close, 0) / 10;
      }
      if (index >= 19) {
        item.sma20 =
          data
            .slice(index - 19, index + 1)
            .reduce((sum, d) => sum + d.close, 0) / 20;
      }
      if (index >= 49) {
        item.sma50 =
          data
            .slice(index - 49, index + 1)
            .reduce((sum, d) => sum + d.close, 0) / 50;
      }
    });

    return data;
  };

  // Helper function to calculate EMA
  const calculateEMA = (data, period, field) => {
    const multiplier = 2 / (period + 1);
    let ema =
      data.slice(0, period).reduce((sum, item) => sum + item[field], 0) /
      period;

    for (let i = period; i < data.length; i++) {
      ema = data[i][field] * multiplier + ema * (1 - multiplier);
    }

    return ema;
  };

  const toggleIndicator = (indicator) => {
    setSelectedIndicators((prev) => ({
      ...prev,
      [indicator]: !prev[indicator],
    }));
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg min-w-48">
          <p className="font-medium text-gray-900 mb-2">{formatDate(label)}</p>
          <div className="space-y-1 text-sm">
            {payload.map((entry, index) => (
              <div key={index} className="flex justify-between items-center">
                <span style={{ color: entry.color }}>{entry.name}:</span>
                <span className="font-medium ml-2">
                  {typeof entry.value === 'number'
                    ? entry.value.toFixed(2)
                    : entry.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    return null;
  };

  if (!scanData || availableStocks.length === 0) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
      >
        <div className="text-center py-12">
          <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No Data Available for Technical Analysis
          </h3>
          <p className="text-gray-600">
            Run your breakout scan to generate technical indicator data.
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
            <Zap className="h-6 w-6" />
            Advanced Technical Indicators
          </h3>
          <p className="text-gray-600 mt-1">
            Professional technical analysis with multiple indicators
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            {['30d', '60d', '90d'].map((period) => (
              <button
                key={period}
                onClick={() => setTimeframe(period)}
                className={`px-3 py-1 text-sm rounded ${
                  timeframe === period
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {period}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stock Selector */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Stock for Technical Analysis:
        </label>
        <select
          value={selectedStock?.symbol || ''}
          onChange={(e) => {
            const stock = availableStocks.find(
              (s) => s.symbol === e.target.value
            );
            setSelectedStock(stock);
          }}
          className="w-full md:w-1/2 border border-gray-300 rounded-lg px-3 py-2"
        >
          <option value="">Choose a stock...</option>
          {availableStocks.slice(0, 20).map((stock) => (
            <option key={stock.symbol} value={stock.symbol}>
              {stock.symbol} - {stock.type} (
              {stock.price_change_pct > 0 ? '+' : ''}
              {stock.price_change_pct?.toFixed(1)}%)
            </option>
          ))}
        </select>
      </div>

      {selectedStock && (
        <>
          {/* Indicator Controls */}
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Technical Indicators
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {Object.entries({
                rsi: 'RSI (14)',
                macd: 'MACD',
                bollinger: 'Bollinger Bands',
                stochastic: 'Stochastic',
                williams: 'Williams %R',
                momentum: 'Momentum',
              }).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => toggleIndicator(key)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                    selectedIndicators[key]
                      ? 'bg-blue-100 text-blue-700 border border-blue-200'
                      : 'bg-white text-gray-600 border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {selectedIndicators[key] ? (
                    <Eye className="h-3 w-3" />
                  ) : (
                    <EyeOff className="h-3 w-3" />
                  )}
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Price Chart with Moving Averages */}
          <div className="mb-6">
            <h4 className="font-medium text-gray-900 mb-3">
              Price Action & Moving Averages
            </h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={technicalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatDate}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    domain={['dataMin - 1', 'dataMax + 1']}
                  />
                  <Tooltip content={<CustomTooltip />} />

                  {/* Price Line */}
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke="#1f2937"
                    strokeWidth={2}
                    dot={false}
                    name="Price"
                  />

                  {/* Moving Averages */}
                  <Line
                    type="monotone"
                    dataKey="sma10"
                    stroke="#3b82f6"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                    name="SMA 10"
                  />
                  <Line
                    type="monotone"
                    dataKey="sma20"
                    stroke="#10b981"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                    name="SMA 20"
                  />
                  <Line
                    type="monotone"
                    dataKey="sma50"
                    stroke="#f59e0b"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                    name="SMA 50"
                  />

                  {/* Bollinger Bands */}
                  {selectedIndicators.bollinger && (
                    <>
                      <Line
                        type="monotone"
                        dataKey="bollingerUpper"
                        stroke="#ef4444"
                        strokeWidth={1}
                        strokeDasharray="2 2"
                        dot={false}
                        name="BB Upper"
                      />
                      <Line
                        type="monotone"
                        dataKey="bollingerLower"
                        stroke="#ef4444"
                        strokeWidth={1}
                        strokeDasharray="2 2"
                        dot={false}
                        name="BB Lower"
                      />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Technical Indicator Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* RSI Chart */}
            {selectedIndicators.rsi && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">RSI (14)</h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={technicalData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={formatDate}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine
                        y={70}
                        stroke="#ef4444"
                        strokeDasharray="2 2"
                      />
                      <ReferenceLine
                        y={30}
                        stroke="#22c55e"
                        strokeDasharray="2 2"
                      />
                      <Line
                        type="monotone"
                        dataKey="rsi"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        dot={false}
                        name="RSI"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-xs text-gray-600 mt-2">
                  Overbought: >70 | Oversold: &lt;30
                </div>
              </div>
            )}

            {/* MACD Chart */}
            {selectedIndicators.macd && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">
                  MACD (12,26,9)
                </h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={technicalData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={formatDate}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={0} stroke="#6b7280" />
                      <Bar
                        dataKey="macdHistogram"
                        fill="#94a3b8"
                        name="MACD Histogram"
                      />
                      <Line
                        type="monotone"
                        dataKey="macd"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                        name="MACD"
                      />
                      <Line
                        type="monotone"
                        dataKey="macdSignal"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                        name="Signal"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Stochastic Oscillator */}
            {selectedIndicators.stochastic && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">
                  Stochastic (14,3,3)
                </h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={technicalData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={formatDate}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine
                        y={80}
                        stroke="#ef4444"
                        strokeDasharray="2 2"
                      />
                      <ReferenceLine
                        y={20}
                        stroke="#22c55e"
                        strokeDasharray="2 2"
                      />
                      <Line
                        type="monotone"
                        dataKey="stochK"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                        name="%K"
                      />
                      <Line
                        type="monotone"
                        dataKey="stochD"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                        name="%D"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Williams %R */}
            {selectedIndicators.williams && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">
                  Williams %R (14)
                </h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={technicalData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={formatDate}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis domain={[-100, 0]} tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine
                        y={-20}
                        stroke="#ef4444"
                        strokeDasharray="2 2"
                      />
                      <ReferenceLine
                        y={-80}
                        stroke="#22c55e"
                        strokeDasharray="2 2"
                      />
                      <Line
                        type="monotone"
                        dataKey="williamsR"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        dot={false}
                        name="Williams %R"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Momentum */}
            {selectedIndicators.momentum && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">
                  Momentum (10)
                </h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={technicalData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={formatDate}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={0} stroke="#6b7280" />
                      <Line
                        type="monotone"
                        dataKey="momentum"
                        stroke="#ec4899"
                        strokeWidth={2}
                        dot={false}
                        name="Momentum"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {/* Current Indicator Values */}
          <div className="mt-6 bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 mb-3">
              Current Indicator Values
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
              {technicalData.length > 0 && (
                <>
                  {selectedIndicators.rsi &&
                    technicalData[technicalData.length - 1].rsi && (
                      <div className="text-center">
                        <div className="font-bold text-purple-600">
                          {technicalData[technicalData.length - 1].rsi.toFixed(
                            1
                          )}
                        </div>
                        <div className="text-gray-600">RSI</div>
                      </div>
                    )}
                  {selectedIndicators.macd &&
                    technicalData[technicalData.length - 1].macd && (
                      <div className="text-center">
                        <div className="font-bold text-blue-600">
                          {technicalData[technicalData.length - 1].macd.toFixed(
                            3
                          )}
                        </div>
                        <div className="text-gray-600">MACD</div>
                      </div>
                    )}
                  {selectedIndicators.stochastic &&
                    technicalData[technicalData.length - 1].stochK && (
                      <div className="text-center">
                        <div className="font-bold text-indigo-600">
                          {technicalData[
                            technicalData.length - 1
                          ].stochK.toFixed(1)}
                        </div>
                        <div className="text-gray-600">Stoch %K</div>
                      </div>
                    )}
                  {selectedIndicators.williams &&
                    technicalData[technicalData.length - 1].williamsR && (
                      <div className="text-center">
                        <div className="font-bold text-orange-600">
                          {technicalData[
                            technicalData.length - 1
                          ].williamsR.toFixed(1)}
                        </div>
                        <div className="text-gray-600">Williams %R</div>
                      </div>
                    )}
                  {selectedIndicators.momentum &&
                    technicalData[technicalData.length - 1].momentum && (
                      <div className="text-center">
                        <div className="font-bold text-pink-600">
                          {technicalData[
                            technicalData.length - 1
                          ].momentum.toFixed(1)}
                          %
                        </div>
                        <div className="text-gray-600">Momentum</div>
                      </div>
                    )}
                </>
              )}
            </div>
          </div>

          {/* Trading Signals */}
          <div className="mt-6 bg-blue-50 rounded-lg p-4">
            <h4 className="font-medium text-blue-900 mb-3 flex items-center gap-2">
              <Target className="h-4 w-4" />
              Technical Analysis Summary
            </h4>
            <div className="text-sm text-blue-800 space-y-2">
              <p>
                <strong>{selectedStock.symbol}</strong> is currently showing a{' '}
                <span
                  className={`font-medium ${
                    selectedStock.type === 'bullish'
                      ? 'text-green-600'
                      : selectedStock.type === 'bearish'
                      ? 'text-red-600'
                      : 'text-blue-600'
                  }`}
                >
                  {selectedStock.type} breakout
                </span>{' '}
                pattern with {selectedStock.volume_ratio?.toFixed(1)}x average
                volume.
              </p>
              {technicalData.length > 0 &&
                technicalData[technicalData.length - 1].rsi && (
                  <p>
                    RSI at{' '}
                    {technicalData[technicalData.length - 1].rsi.toFixed(1)}{' '}
                    indicates the stock is{' '}
                    {technicalData[technicalData.length - 1].rsi > 70
                      ? 'overbought'
                      : technicalData[technicalData.length - 1].rsi < 30
                      ? 'oversold'
                      : 'in neutral territory'}
                    .
                  </p>
                )}
              <p className="text-xs text-blue-600 mt-2">
                * This analysis is for educational purposes only and should not
                be considered as investment advice.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default AdvancedTechnicalIndicators;
