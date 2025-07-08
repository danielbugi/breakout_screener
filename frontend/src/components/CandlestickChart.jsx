import React, { useState, useEffect, useMemo } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Calendar,
  DollarSign,
  BarChart3,
  Settings,
  Eye,
  EyeOff,
} from 'lucide-react';

const CandlestickChart = ({ symbol, stockData, className = '' }) => {
  const [chartData, setChartData] = useState([]);
  const [timeframe, setTimeframe] = useState('20d');
  const [showVolume, setShowVolume] = useState(true);
  const [showIndicators, setShowIndicators] = useState({
    sma20: true,
    sma50: false,
    rsi: false,
    donchian: true,
  });

  useEffect(() => {
    generateAdvancedChartData();
  }, [symbol, stockData, timeframe]);

  const generateAdvancedChartData = () => {
    if (!stockData) return;

    const days = timeframe === '20d' ? 40 : timeframe === '50d' ? 70 : 120;
    const data = [];
    const currentPrice = stockData.close_price;
    const high = stockData.donchian_high;
    const low = stockData.donchian_low;
    const mid = stockData.donchian_mid;

    // Generate realistic OHLC data
    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);

      let basePrice;
      let volatility = 0.02; // 2% daily volatility

      if (i === 0) {
        // Current day - actual breakout
        basePrice = currentPrice;
      } else if (i <= 5) {
        // Recent days - building momentum
        const progress = (5 - i) / 5;
        if (stockData.bullish_breakout) {
          basePrice = low + (high - low) * (0.2 + progress * 0.8);
          volatility = 0.015 + progress * 0.01; // Increasing volatility toward breakout
        } else if (stockData.bearish_breakout) {
          basePrice = high - (high - low) * (0.2 + progress * 0.8);
          volatility = 0.015 + progress * 0.01;
        } else {
          basePrice = mid + (Math.random() - 0.5) * (high - low) * 0.4;
        }
      } else {
        // Historical data
        basePrice = low + (high - low) * (0.3 + Math.random() * 0.4);
        volatility = 0.015 + Math.random() * 0.01;
      }

      // Generate OHLC
      const openPrice =
        basePrice + (Math.random() - 0.5) * basePrice * volatility;
      const closePrice =
        basePrice + (Math.random() - 0.5) * basePrice * volatility;
      const highPrice =
        Math.max(openPrice, closePrice) +
        Math.random() * basePrice * volatility * 0.5;
      const lowPrice =
        Math.min(openPrice, closePrice) -
        Math.random() * basePrice * volatility * 0.5;

      // Ensure breakout day is accurate
      if (i === 0) {
        const finalClose = currentPrice;
        const finalOpen =
          finalClose + (Math.random() - 0.5) * finalClose * 0.01;
        data.push({
          date: date.toISOString().split('T')[0],
          open: Math.max(finalOpen, low * 0.99),
          high: stockData.bullish_breakout
            ? Math.max(finalClose, high * 1.001)
            : Math.max(finalOpen, finalClose) + finalClose * 0.005,
          low: stockData.bearish_breakout
            ? Math.min(finalClose, low * 0.999)
            : Math.min(finalOpen, finalClose) - finalClose * 0.005,
          close: finalClose,
          volume: Math.random() * 2000000 + 1000000,
          donchian_high: high,
          donchian_low: low,
          donchian_mid: mid,
          isBreakout: true,
        });
      } else {
        data.push({
          date: date.toISOString().split('T')[0],
          open: Math.max(openPrice, low * 0.95),
          high: Math.min(highPrice, high * 1.05),
          low: Math.max(lowPrice, low * 0.95),
          close: Math.max(closePrice, low * 0.95),
          volume: Math.random() * 1500000 + 500000,
          donchian_high: high,
          donchian_low: low,
          donchian_mid: mid,
          isBreakout: false,
        });
      }
    }

    // Calculate technical indicators
    calculateTechnicalIndicators(data);
    setChartData(data);
  };

  const calculateTechnicalIndicators = (data) => {
    // Simple Moving Averages
    data.forEach((item, index) => {
      // SMA 20
      if (index >= 19) {
        const sum20 = data
          .slice(index - 19, index + 1)
          .reduce((acc, d) => acc + d.close, 0);
        item.sma20 = sum20 / 20;
      }

      // SMA 50
      if (index >= 49) {
        const sum50 = data
          .slice(index - 49, index + 1)
          .reduce((acc, d) => acc + d.close, 0);
        item.sma50 = sum50 / 50;
      }

      // RSI calculation (simplified)
      if (index >= 14) {
        const period = 14;
        const changes = data
          .slice(index - period + 1, index + 1)
          .map((d, i, arr) => (i === 0 ? 0 : d.close - arr[i - 1].close));
        const gains =
          changes.filter((c) => c > 0).reduce((acc, c) => acc + c, 0) / period;
        const losses =
          Math.abs(
            changes.filter((c) => c < 0).reduce((acc, c) => acc + c, 0)
          ) / period;
        const rs = gains / (losses || 1);
        item.rsi = 100 - 100 / (1 + rs);
      }
    });
  };

  // Custom Candlestick component
  const Candlestick = (props) => {
    const { payload, x, y, width, height } = props;
    if (!payload) return null;

    const { open, high, low, close } = payload;
    const isUp = close > open;
    const color = isUp ? '#22c55e' : '#ef4444';
    const bodyHeight = (Math.abs(close - open) * height) / (high - low);
    const bodyY = y + ((high - Math.max(open, close)) * height) / (high - low);

    return (
      <g>
        {/* Wick */}
        <line
          x1={x + width / 2}
          y1={y}
          x2={x + width / 2}
          y2={y + height}
          stroke={color}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={x + width * 0.25}
          y={bodyY}
          width={width * 0.5}
          height={Math.max(bodyHeight, 1)}
          fill={isUp ? color : '#ffffff'}
          stroke={color}
          strokeWidth={1}
        />
        {/* Breakout indicator */}
        {payload.isBreakout && (
          <circle
            cx={x + width / 2}
            cy={bodyY + bodyHeight / 2}
            r={3}
            fill="#f59e0b"
            stroke="#ffffff"
            strokeWidth={1}
          />
        )}
      </g>
    );
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      if (!data) return null;

      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg min-w-48">
          <p className="font-medium text-gray-900 mb-2">{label}</p>
          <div className="space-y-1 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <span>Open:</span>
              <span className="font-medium">${data.open?.toFixed(2)}</span>
              <span>High:</span>
              <span className="font-medium text-green-600">
                ${data.high?.toFixed(2)}
              </span>
              <span>Low:</span>
              <span className="font-medium text-red-600">
                ${data.low?.toFixed(2)}
              </span>
              <span>Close:</span>
              <span className="font-medium">${data.close?.toFixed(2)}</span>
            </div>
            <hr className="my-2" />
            <div className="grid grid-cols-2 gap-2">
              <span>Volume:</span>
              <span className="font-medium">
                {(data.volume / 1000000).toFixed(1)}M
              </span>
              {data.rsi && (
                <>
                  <span>RSI:</span>
                  <span className="font-medium">{data.rsi.toFixed(1)}</span>
                </>
              )}
            </div>
            {data.isBreakout && (
              <p className="text-orange-600 font-medium text-center mt-2">
                🚀 Breakout Point!
              </p>
            )}
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

  const formatVolume = (value) => {
    return `${(value / 1000000).toFixed(1)}M`;
  };

  const toggleIndicator = (indicator) => {
    setShowIndicators((prev) => ({
      ...prev,
      [indicator]: !prev[indicator],
    }));
  };

  if (!stockData || chartData.length === 0) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
      >
        <div className="flex items-center justify-center h-96 text-gray-500">
          <Activity className="h-8 w-8 mr-2" />
          Loading advanced chart data...
        </div>
      </div>
    );
  }

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
    >
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            {symbol} - Advanced Technical Analysis
          </h3>
          <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
            <span className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              {stockData.date}
            </span>
            <span
              className={`font-medium ${
                stockData.price_change_pct > 0
                  ? 'text-green-600'
                  : 'text-red-600'
              }`}
            >
              {stockData.price_change_pct > 0 ? '+' : ''}
              {stockData.price_change_pct?.toFixed(2)}%
            </span>
            <span className="flex items-center gap-1">
              {stockData.bullish_breakout ? (
                <>
                  <TrendingUp className="h-4 w-4 text-green-600" />
                  <span className="text-green-600 font-medium">
                    Bullish Breakout
                  </span>
                </>
              ) : stockData.bearish_breakout ? (
                <>
                  <TrendingDown className="h-4 w-4 text-red-600" />
                  <span className="text-red-600 font-medium">
                    Bearish Breakout
                  </span>
                </>
              ) : (
                <>
                  <Activity className="h-4 w-4 text-blue-600" />
                  <span className="text-blue-600 font-medium">
                    Near Breakout
                  </span>
                </>
              )}
            </span>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Timeframe Selector */}
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            {['20d', '50d', '100d'].map((period) => (
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

          {/* Indicator Controls */}
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4 text-gray-500" />
            <button
              onClick={() => setShowVolume(!showVolume)}
              className={`text-xs px-2 py-1 rounded ${
                showVolume
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              Volume
            </button>
            <button
              onClick={() => toggleIndicator('sma20')}
              className={`text-xs px-2 py-1 rounded ${
                showIndicators.sma20
                  ? 'bg-green-100 text-green-700'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              SMA20
            </button>
            <button
              onClick={() => toggleIndicator('sma50')}
              className={`text-xs px-2 py-1 rounded ${
                showIndicators.sma50
                  ? 'bg-purple-100 text-purple-700'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              SMA50
            </button>
            <button
              onClick={() => toggleIndicator('rsi')}
              className={`text-xs px-2 py-1 rounded ${
                showIndicators.rsi
                  ? 'bg-orange-100 text-orange-700'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              RSI
            </button>
          </div>
        </div>
      </div>

      {/* Main Price Chart */}
      <div className="h-96 mb-4">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              domain={['dataMin - 1', 'dataMax + 1']}
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Donchian Channels */}
            {showIndicators.donchian && (
              <>
                <Line
                  type="monotone"
                  dataKey="donchian_high"
                  stroke="#22c55e"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Donchian High"
                />
                <Line
                  type="monotone"
                  dataKey="donchian_low"
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Donchian Low"
                />
                <Line
                  type="monotone"
                  dataKey="donchian_mid"
                  stroke="#6b7280"
                  strokeWidth={1}
                  strokeDasharray="2 2"
                  dot={false}
                  name="Donchian Mid"
                />
              </>
            )}

            {/* Moving Averages */}
            {showIndicators.sma20 && (
              <Line
                type="monotone"
                dataKey="sma20"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="SMA 20"
              />
            )}

            {showIndicators.sma50 && (
              <Line
                type="monotone"
                dataKey="sma50"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={false}
                name="SMA 50"
              />
            )}

            {/* Candlestick representation using custom shapes */}
            <Bar dataKey="high" fill="transparent" shape={<Candlestick />} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Volume Chart */}
      {showVolume && (
        <div className="h-24 mb-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fontSize: 10 }}
              />
              <YAxis tick={{ fontSize: 10 }} tickFormatter={formatVolume} />
              <Bar dataKey="volume" fill="#94a3b8" name="Volume" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* RSI Indicator */}
      {showIndicators.rsi && (
        <div className="h-20 mb-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fontSize: 10 }}
              />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
              <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="2 2" />
              <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="2 2" />
              <Line
                type="monotone"
                dataKey="rsi"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                name="RSI"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Trading Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-gray-50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-gray-900">
            ${stockData.close_price?.toFixed(2)}
          </div>
          <div className="text-sm text-gray-600">Current Price</div>
        </div>
        <div className="bg-green-50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-green-700">
            ${chartData[chartData.length - 1]?.high?.toFixed(2)}
          </div>
          <div className="text-sm text-green-600">Day High</div>
        </div>
        <div className="bg-red-50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-red-700">
            ${chartData[chartData.length - 1]?.low?.toFixed(2)}
          </div>
          <div className="text-sm text-red-600">Day Low</div>
        </div>
        <div className="bg-blue-50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-blue-700">
            {formatVolume(chartData[chartData.length - 1]?.volume || 0)}
          </div>
          <div className="text-sm text-blue-600">Volume</div>
        </div>
      </div>

      {/* Technical Analysis Summary */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-2">Technical Analysis</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-gray-700 leading-relaxed">
              <strong>Price Action:</strong> {symbol} is currently{' '}
              {stockData.bullish_breakout
                ? 'breaking above'
                : stockData.bearish_breakout
                ? 'breaking below'
                : 'trading within'}{' '}
              the Donchian channel. The {stockData.volume_ratio?.toFixed(1)}x
              volume indicates{' '}
              {stockData.volume_ratio > 2
                ? 'strong'
                : stockData.volume_ratio > 1.5
                ? 'moderate'
                : 'weak'}{' '}
              conviction.
            </p>
          </div>
          <div>
            <p className="text-gray-700 leading-relaxed">
              <strong>Technical Signals:</strong>{' '}
              {showIndicators.sma20 && chartData[chartData.length - 1]?.sma20
                ? `Price is ${
                    stockData.close_price >
                    chartData[chartData.length - 1].sma20
                      ? 'above'
                      : 'below'
                  } SMA20. `
                : ''}
              Position within channel:{' '}
              {stockData.price_position_pct?.toFixed(1)}%.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CandlestickChart;
