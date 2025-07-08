// src/components/VolumeProfileAnalysis.jsx
import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ComposedChart,
  Line,
  Area,
  AreaChart,
} from 'recharts';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  Eye,
  Settings,
  BarChart3,
} from 'lucide-react';

const VolumeProfileAnalysis = ({ scanData, className = '' }) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('daily');
  const [analysisType, setAnalysisType] = useState('distribution');
  const [showVolumePrice, setShowVolumePrice] = useState(true);

  // Process volume data from scan results
  const volumeAnalysis = useMemo(() => {
    if (!scanData || !scanData.breakouts) return null;

    const allBreakouts = [
      ...(scanData.breakouts.bullish || []),
      ...(scanData.breakouts.bearish || []),
      ...(scanData.breakouts.near_bullish || []),
      ...(scanData.breakouts.near_bearish || []),
    ];

    if (allBreakouts.length === 0) return null;

    // Volume distribution analysis
    const volumeRanges = [
      {
        range: '1.0-1.5x',
        min: 1.0,
        max: 1.5,
        count: 0,
        totalValue: 0,
        color: '#e5e7eb',
      },
      {
        range: '1.5-2.0x',
        min: 1.5,
        max: 2.0,
        count: 0,
        totalValue: 0,
        color: '#d1d5db',
      },
      {
        range: '2.0-3.0x',
        min: 2.0,
        max: 3.0,
        count: 0,
        totalValue: 0,
        color: '#9ca3af',
      },
      {
        range: '3.0-5.0x',
        min: 3.0,
        max: 5.0,
        count: 0,
        totalValue: 0,
        color: '#6b7280',
      },
      {
        range: '5.0x+',
        min: 5.0,
        max: 100,
        count: 0,
        totalValue: 0,
        color: '#374151',
      },
    ];

    // Price level analysis (divide into price ranges)
    const priceRanges = Array.from({ length: 10 }, (_, i) => ({
      range: `$${(i * 50).toFixed(0)}-${((i + 1) * 50).toFixed(0)}`,
      min: i * 50,
      max: (i + 1) * 50,
      volume: 0,
      avgChange: 0,
      count: 0,
      bullishCount: 0,
      bearishCount: 0,
    }));

    // Categorize breakouts
    const volumeBreakouts = {
      high: [],
      medium: [],
      low: [],
    };

    allBreakouts.forEach((stock) => {
      const vol = stock.volume_ratio || 1;
      const price = stock.close_price || 50;

      // Volume range classification
      volumeRanges.forEach((range) => {
        if (vol >= range.min && vol < range.max) {
          range.count++;
          range.totalValue += Math.abs(stock.price_change_pct || 0);
        }
      });

      // Price range classification
      priceRanges.forEach((range) => {
        if (price >= range.min && price < range.max) {
          range.volume += vol;
          range.avgChange += Math.abs(stock.price_change_pct || 0);
          range.count++;
          if (stock.bullish_breakout) range.bullishCount++;
          if (stock.bearish_breakout) range.bearishCount++;
        }
      });

      // Volume breakout classification
      if (vol >= 3.0) {
        volumeBreakouts.high.push(stock);
      } else if (vol >= 1.5) {
        volumeBreakouts.medium.push(stock);
      } else {
        volumeBreakouts.low.push(stock);
      }
    });

    // Calculate averages for price ranges
    priceRanges.forEach((range) => {
      if (range.count > 0) {
        range.avgChange = range.avgChange / range.count;
        range.volume = range.volume / range.count;
      }
    });

    // Calculate average values for volume ranges
    volumeRanges.forEach((range) => {
      if (range.count > 0) {
        range.avgChange = range.totalValue / range.count;
      }
    });

    return {
      volumeRanges: volumeRanges.filter((r) => r.count > 0),
      priceRanges: priceRanges.filter((r) => r.count > 0),
      volumeBreakouts,
      totalBreakouts: allBreakouts.length,
      stats: {
        avgVolume:
          allBreakouts.reduce((sum, s) => sum + (s.volume_ratio || 1), 0) /
          allBreakouts.length,
        maxVolume: Math.max(...allBreakouts.map((s) => s.volume_ratio || 1)),
        highVolumeCount: volumeBreakouts.high.length,
        mediumVolumeCount: volumeBreakouts.medium.length,
        lowVolumeCount: volumeBreakouts.low.length,
      },
    };
  }, [scanData]);

  // Custom tooltip for volume analysis
  const VolumeTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900 mb-2">{label}</p>
          <div className="space-y-1 text-sm">
            <p>
              Count: <span className="font-medium">{data.count}</span>
            </p>
            {data.avgChange && (
              <p>
                Avg Change:{' '}
                <span className="font-medium">
                  {data.avgChange.toFixed(1)}%
                </span>
              </p>
            )}
            {data.volume && (
              <p>
                Avg Volume:{' '}
                <span className="font-medium">{data.volume.toFixed(1)}x</span>
              </p>
            )}
            {data.bullishCount !== undefined && (
              <div className="mt-2 pt-2 border-t border-gray-200">
                <p className="text-green-600">Bullish: {data.bullishCount}</p>
                <p className="text-red-600">Bearish: {data.bearishCount}</p>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  if (!volumeAnalysis) {
    return (
      <div
        className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}
      >
        <div className="text-center py-12">
          <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No Volume Data Available
          </h3>
          <p className="text-gray-600">
            Run your breakout scan to generate volume profile analysis.
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
            Volume Profile Analysis
          </h3>
          <p className="text-gray-600 mt-1">
            Analyze volume distribution across {volumeAnalysis.totalBreakouts}{' '}
            breakouts
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <select
            value={analysisType}
            onChange={(e) => setAnalysisType(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          >
            <option value="distribution">Volume Distribution</option>
            <option value="price-levels">Price Level Analysis</option>
            <option value="correlation">Volume-Price Correlation</option>
          </select>

          <button
            onClick={() => setShowVolumePrice(!showVolumePrice)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
              showVolumePrice
                ? 'bg-blue-100 text-blue-700'
                : 'bg-gray-100 text-gray-600'
            }`}
          >
            <Eye className="h-4 w-4" />
            Price Overlay
          </button>
        </div>
      </div>

      {/* Key Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">
            {volumeAnalysis.stats.avgVolume.toFixed(1)}x
          </div>
          <div className="text-sm text-blue-700">Average Volume</div>
        </div>
        <div className="bg-green-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-600">
            {volumeAnalysis.stats.highVolumeCount}
          </div>
          <div className="text-sm text-green-700">High Volume (3x+)</div>
        </div>
        <div className="bg-yellow-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-yellow-600">
            {volumeAnalysis.stats.mediumVolumeCount}
          </div>
          <div className="text-sm text-yellow-700">Medium Volume (1.5-3x)</div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-gray-600">
            {volumeAnalysis.stats.maxVolume.toFixed(1)}x
          </div>
          <div className="text-sm text-gray-700">Peak Volume</div>
        </div>
      </div>

      {/* Main Chart */}
      <div className="h-80 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          {analysisType === 'distribution' ? (
            <BarChart data={volumeAnalysis.volumeRanges}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="range"
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                label={{ value: 'Count', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip content={<VolumeTooltip />} />
              <Bar dataKey="count" name="Breakout Count">
                {volumeAnalysis.volumeRanges.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          ) : analysisType === 'price-levels' ? (
            <ComposedChart data={volumeAnalysis.priceRanges}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="range"
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis
                yAxisId="volume"
                tick={{ fontSize: 12 }}
                label={{
                  value: 'Avg Volume Ratio',
                  angle: -90,
                  position: 'insideLeft',
                }}
              />
              <YAxis
                yAxisId="change"
                orientation="right"
                tick={{ fontSize: 12 }}
                label={{
                  value: 'Avg Change %',
                  angle: 90,
                  position: 'insideRight',
                }}
              />
              <Tooltip content={<VolumeTooltip />} />
              <Bar
                yAxisId="volume"
                dataKey="volume"
                fill="#3b82f6"
                name="Avg Volume"
              />
              {showVolumePrice && (
                <Line
                  yAxisId="change"
                  type="monotone"
                  dataKey="avgChange"
                  stroke="#ef4444"
                  strokeWidth={3}
                  name="Avg Change %"
                />
              )}
            </ComposedChart>
          ) : (
            <AreaChart data={volumeAnalysis.priceRanges}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="range"
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip content={<VolumeTooltip />} />
              <Area
                type="monotone"
                dataKey="volume"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.6}
                name="Volume Pattern"
              />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Volume Breakout Categories */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* High Volume Breakouts */}
        <div className="bg-green-50 rounded-lg p-4">
          <h4 className="font-medium text-green-800 mb-3 flex items-center gap-2">
            <Zap className="h-4 w-4" />
            High Volume (3x+)
          </h4>
          <div className="space-y-2">
            <div className="text-2xl font-bold text-green-600">
              {volumeAnalysis.volumeBreakouts.high.length}
            </div>
            <p className="text-sm text-green-700">
              {(
                (volumeAnalysis.volumeBreakouts.high.length /
                  volumeAnalysis.totalBreakouts) *
                100
              ).toFixed(1)}
              % of total
            </p>
            {volumeAnalysis.volumeBreakouts.high.slice(0, 3).map((stock) => (
              <div key={stock.symbol} className="text-xs">
                <span className="font-medium">{stock.symbol}</span>
                <span className="text-green-600 ml-2">
                  {stock.volume_ratio?.toFixed(1)}x
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Medium Volume Breakouts */}
        <div className="bg-yellow-50 rounded-lg p-4">
          <h4 className="font-medium text-yellow-800 mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Medium Volume (1.5-3x)
          </h4>
          <div className="space-y-2">
            <div className="text-2xl font-bold text-yellow-600">
              {volumeAnalysis.volumeBreakouts.medium.length}
            </div>
            <p className="text-sm text-yellow-700">
              {(
                (volumeAnalysis.volumeBreakouts.medium.length /
                  volumeAnalysis.totalBreakouts) *
                100
              ).toFixed(1)}
              % of total
            </p>
            {volumeAnalysis.volumeBreakouts.medium.slice(0, 3).map((stock) => (
              <div key={stock.symbol} className="text-xs">
                <span className="font-medium">{stock.symbol}</span>
                <span className="text-yellow-600 ml-2">
                  {stock.volume_ratio?.toFixed(1)}x
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Low Volume Breakouts */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-800 mb-3 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Low Volume (1-1.5x)
          </h4>
          <div className="space-y-2">
            <div className="text-2xl font-bold text-gray-600">
              {volumeAnalysis.volumeBreakouts.low.length}
            </div>
            <p className="text-sm text-gray-700">
              {(
                (volumeAnalysis.volumeBreakouts.low.length /
                  volumeAnalysis.totalBreakouts) *
                100
              ).toFixed(1)}
              % of total
            </p>
            {volumeAnalysis.volumeBreakouts.low.slice(0, 3).map((stock) => (
              <div key={stock.symbol} className="text-xs">
                <span className="font-medium">{stock.symbol}</span>
                <span className="text-gray-600 ml-2">
                  {stock.volume_ratio?.toFixed(1)}x
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Analysis Insights */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-3">
          Volume Profile Insights
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h5 className="font-medium text-gray-800 mb-2">
              Volume Distribution
            </h5>
            <ul className="space-y-1 text-gray-700">
              <li>
                •{' '}
                <strong>
                  {(
                    (volumeAnalysis.stats.highVolumeCount /
                      volumeAnalysis.totalBreakouts) *
                    100
                  ).toFixed(1)}
                  %
                </strong>
                of breakouts show high conviction (3x+ volume)
              </li>
              <li>
                • Peak volume ratio:{' '}
                <strong>{volumeAnalysis.stats.maxVolume.toFixed(1)}x</strong>{' '}
                average
              </li>
              <li>
                • Most breakouts occur in the{' '}
                <strong>
                  {
                    volumeAnalysis.volumeRanges.reduce((max, range) =>
                      range.count > max.count ? range : max
                    ).range
                  }
                </strong>{' '}
                range
              </li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium text-gray-800 mb-2">
              Trading Implications
            </h5>
            <ul className="space-y-1 text-gray-700">
              <li>• High volume breakouts show stronger momentum</li>
              <li>• Volume expansion often precedes major moves</li>
              <li>• Low volume breakouts may lack follow-through</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VolumeProfileAnalysis;
