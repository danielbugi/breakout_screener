import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Calendar,
  DollarSign,
} from 'lucide-react';

const StockChart = ({ symbol, stockData, className = '' }) => {
  const [chartData, setChartData] = useState([]);
  const [timeframe, setTimeframe] = useState('20d');

  useEffect(() => {
    // Generate mock historical data for demonstration
    // In a real app, this would fetch actual historical price data
    generateMockChartData();
  }, [symbol, stockData, timeframe]);

  const generateMockChartData = () => {
    if (!stockData) return;

    const days = timeframe === '20d' ? 30 : timeframe === '50d' ? 60 : 90;
    const data = [];
    const currentPrice = stockData.close_price;
    const high = stockData.donchian_high;
    const low = stockData.donchian_low;
    const mid = stockData.donchian_mid;

    // Generate historical data leading up to the breakout
    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);

      // Create realistic price movement within the channel
      let price;
      if (i === 0) {
        // Current day - actual breakout price
        price = currentPrice;
      } else if (i <= 5) {
        // Recent days - moving toward breakout
        const progress = (5 - i) / 5;
        if (stockData.bullish_breakout) {
          price =
            low +
            (high - low) * (0.3 + progress * 0.7) +
            (Math.random() - 0.5) * (high - low) * 0.1;
        } else if (stockData.bearish_breakout) {
          price =
            high -
            (high - low) * (0.3 + progress * 0.7) +
            (Math.random() - 0.5) * (high - low) * 0.1;
        } else {
          price = mid + (Math.random() - 0.5) * (high - low) * 0.6;
        }
      } else {
        // Historical data - random movement within channel
        price = low + (high - low) * Math.random();
      }

      // Add some realistic volatility
      price += (Math.random() - 0.5) * currentPrice * 0.02;

      data.push({
        date: date.toISOString().split('T')[0],
        price: Math.max(price, low * 0.95), // Ensure price doesn't go too far below low
        high: high,
        low: low,
        mid: mid,
        volume: Math.random() * 2000000 + 500000, // Random volume
        isBreakout:
          i === 0 && (stockData.bullish_breakout || stockData.bearish_breakout),
      });
    }

    setChartData(data);
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900 mb-2">{label}</p>
          <div className="space-y-1 text-sm">
            <p className="text-blue-600">Price: ${data.price?.toFixed(2)}</p>
            <p className="text-green-600">
              Channel High: ${data.high?.toFixed(2)}
            </p>
            <p className="text-red-600">Channel Low: ${data.low?.toFixed(2)}</p>
            <p className="text-gray-600">
              Channel Mid: ${data.mid?.toFixed(2)}
            </p>
            {data.isBreakout && (
              <p className="text-orange-600 font-medium">🚀 Breakout Point!</p>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  const getBreakoutColor = () => {
    if (stockData?.bullish_breakout) return '#22c55e';
    if (stockData?.bearish_breakout) return '#ef4444';
    return '#3b82f6';
  };

  const getBreakoutDirection = () => {
    if (stockData?.bullish_breakout) return 'above';
    if (stockData?.bearish_breakout) return 'below';
    return 'within';
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  if (!stockData || chartData.length === 0) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
      >
        <div className="flex items-center justify-center h-64 text-gray-500">
          <Activity className="h-8 w-8 mr-2" />
          Loading chart data...
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
            <DollarSign className="h-5 w-5" />
            {symbol} - Donchian Channel Analysis
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
      </div>

      {/* Price Chart with Donchian Channels */}
      <div className="h-80 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
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

            {/* Donchian Channel High */}
            <Line
              type="monotone"
              dataKey="high"
              stroke="#22c55e"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Channel High"
            />

            {/* Donchian Channel Low */}
            <Line
              type="monotone"
              dataKey="low"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Channel Low"
            />

            {/* Donchian Channel Mid */}
            <Line
              type="monotone"
              dataKey="mid"
              stroke="#6b7280"
              strokeWidth={1}
              strokeDasharray="2 2"
              dot={false}
              name="Channel Mid"
            />

            {/* Price Line */}
            <Line
              type="monotone"
              dataKey="price"
              stroke={getBreakoutColor()}
              strokeWidth={3}
              dot={(props) => {
                const { payload } = props;
                if (payload.isBreakout) {
                  return (
                    <circle
                      cx={props.cx}
                      cy={props.cy}
                      r={6}
                      fill={getBreakoutColor()}
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  );
                }
                return null;
              }}
              name="Price"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-gray-50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-gray-900">
            ${stockData.close_price?.toFixed(2)}
          </div>
          <div className="text-sm text-gray-600">Current Price</div>
        </div>
        <div className="bg-green-50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-green-700">
            ${stockData.donchian_high?.toFixed(2)}
          </div>
          <div className="text-sm text-green-600">Channel High</div>
        </div>
        <div className="bg-red-50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-red-700">
            ${stockData.donchian_low?.toFixed(2)}
          </div>
          <div className="text-sm text-red-600">Channel Low</div>
        </div>
        <div className="bg-blue-50 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-blue-700">
            {stockData.volume_ratio?.toFixed(1)}x
          </div>
          <div className="text-sm text-blue-600">Volume Ratio</div>
        </div>
      </div>

      {/* Analysis Summary */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-2">Analysis Summary</h4>
        <p className="text-sm text-gray-700 leading-relaxed">
          {symbol} is currently trading{' '}
          <span
            className={`font-medium ${
              stockData.bullish_breakout
                ? 'text-green-600'
                : stockData.bearish_breakout
                ? 'text-red-600'
                : 'text-blue-600'
            }`}
          >
            {getBreakoutDirection()} the {stockData.period || 20}-day Donchian
            channel
          </span>{' '}
          at ${stockData.close_price?.toFixed(2)}. The stock shows{' '}
          <span className="font-medium">
            {stockData.volume_ratio?.toFixed(1)}x normal volume
          </span>{' '}
          and a{' '}
          <span
            className={`font-medium ${
              stockData.price_change_pct > 0 ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {stockData.price_change_pct?.toFixed(1)}% price change
          </span>
          . Position within channel: {stockData.price_position_pct?.toFixed(1)}
          %.
        </p>
      </div>
    </div>
  );
};

export default StockChart;
