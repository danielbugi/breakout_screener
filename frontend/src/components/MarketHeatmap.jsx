import React, { useState, useMemo } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Filter,
  Search,
  Eye,
} from 'lucide-react';

const MarketHeatmap = ({ scanData, onStockSelect }) => {
  const [sortBy, setSortBy] = useState('change');
  const [filterType, setFilterType] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Combine and process all breakout data
  const heatmapData = useMemo(() => {
    if (!scanData) return [];

    const allBreakouts = [];

    // Add all breakout types
    scanData.breakouts.bullish?.forEach((stock) => {
      allBreakouts.push({
        ...stock,
        type: 'bullish',
        typeLabel: 'Bullish',
        intensity: Math.abs(stock.price_change_pct) * stock.volume_ratio,
      });
    });

    scanData.breakouts.bearish?.forEach((stock) => {
      allBreakouts.push({
        ...stock,
        type: 'bearish',
        typeLabel: 'Bearish',
        intensity: Math.abs(stock.price_change_pct) * stock.volume_ratio,
      });
    });

    scanData.breakouts.near_bullish?.forEach((stock) => {
      allBreakouts.push({
        ...stock,
        type: 'near_bullish',
        typeLabel: 'Near Bull',
        intensity: (stock.price_position_pct / 100) * stock.volume_ratio,
      });
    });

    scanData.breakouts.near_bearish?.forEach((stock) => {
      allBreakouts.push({
        ...stock,
        type: 'near_bearish',
        typeLabel: 'Near Bear',
        intensity:
          ((100 - stock.price_position_pct) / 100) * stock.volume_ratio,
      });
    });

    // Filter by search term
    let filtered = allBreakouts;
    if (searchTerm) {
      filtered = filtered.filter((stock) =>
        stock.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Filter by type
    if (filterType !== 'all') {
      filtered = filtered.filter((stock) => stock.type === filterType);
    }

    // Sort data
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'change':
          return Math.abs(b.price_change_pct) - Math.abs(a.price_change_pct);
        case 'volume':
          return b.volume_ratio - a.volume_ratio;
        case 'intensity':
          return b.intensity - a.intensity;
        case 'symbol':
          return a.symbol.localeCompare(b.symbol);
        default:
          return 0;
      }
    });

    return filtered;
  }, [scanData, sortBy, filterType, searchTerm]);

  // Get color intensity based on performance and type
  const getHeatmapColor = (stock) => {
    const absChange = Math.abs(stock.price_change_pct);
    const volumeMultiplier = Math.min(stock.volume_ratio / 3, 1); // Cap at 3x volume for color scaling
    const intensity = (absChange / 10) * volumeMultiplier; // Normalize to 0-1 range

    let baseColor, opacity;

    switch (stock.type) {
      case 'bullish':
        baseColor = '34, 197, 94'; // Green
        opacity = Math.min(0.3 + intensity * 0.7, 1);
        break;
      case 'bearish':
        baseColor = '239, 68, 68'; // Red
        opacity = Math.min(0.3 + intensity * 0.7, 1);
        break;
      case 'near_bullish':
        baseColor = '59, 130, 246'; // Blue
        opacity = Math.min(0.2 + (stock.price_position_pct / 100) * 0.6, 0.8);
        break;
      case 'near_bearish':
        baseColor = '245, 158, 11'; // Yellow/Orange
        opacity = Math.min(
          0.2 + ((100 - stock.price_position_pct) / 100) * 0.6,
          0.8
        );
        break;
      default:
        baseColor = '156, 163, 175'; // Gray
        opacity = 0.3;
    }

    return `rgba(${baseColor}, ${opacity})`;
  };

  // Get text color for contrast
  const getTextColor = (stock) => {
    const absChange = Math.abs(stock.price_change_pct);
    const intensity = absChange / 10;

    if (intensity > 0.5 || stock.volume_ratio > 2) {
      return 'text-white';
    }
    return stock.type === 'bullish' || stock.type === 'bearish'
      ? 'text-gray-800'
      : 'text-gray-700';
  };

  // Get icon for breakout type
  const getTypeIcon = (type) => {
    switch (type) {
      case 'bullish':
        return <TrendingUp className="h-3 w-3" />;
      case 'bearish':
        return <TrendingDown className="h-3 w-3" />;
      case 'near_bullish':
        return <TrendingUp className="h-3 w-3 opacity-60" />;
      case 'near_bearish':
        return <TrendingDown className="h-3 w-3 opacity-60" />;
      default:
        return <Activity className="h-3 w-3" />;
    }
  };

  if (!scanData || heatmapData.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
        <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          No Breakout Data Available
        </h3>
        <p className="text-gray-600">
          Run your automation to generate breakout data for the heatmap.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <Activity className="h-6 w-6" />
            Market Heatmap
          </h3>
          <p className="text-gray-600 mt-1">
            Visual overview of {heatmapData.length} breakout signals
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="h-4 w-4 absolute left-3 top-2.5 text-gray-400" />
            <input
              type="text"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Filter */}
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="all">All Types</option>
            <option value="bullish">Bullish Only</option>
            <option value="bearish">Bearish Only</option>
            <option value="near_bullish">Near Bullish</option>
            <option value="near_bearish">Near Bearish</option>
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="change">By Change %</option>
            <option value="volume">By Volume</option>
            <option value="intensity">By Intensity</option>
            <option value="symbol">By Symbol</option>
          </select>
        </div>
      </div>

      {/* Legend */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Legend</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: 'rgba(34, 197, 94, 0.7)' }}
            ></div>
            <span>Bullish Breakout</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: 'rgba(239, 68, 68, 0.7)' }}
            ></div>
            <span>Bearish Breakout</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: 'rgba(59, 130, 246, 0.7)' }}
            ></div>
            <span>Near Bullish</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: 'rgba(245, 158, 11, 0.7)' }}
            ></div>
            <span>Near Bearish</span>
          </div>
        </div>
        <p className="text-xs text-gray-600 mt-2">
          Color intensity represents the combination of price change magnitude
          and volume ratio.
        </p>
      </div>

      {/* Heatmap Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-2">
        {heatmapData.map((stock, index) => (
          <div
            key={`${stock.symbol}-${index}`}
            className={`
              relative p-3 rounded-lg border cursor-pointer transition-all duration-200 hover:scale-105 hover:shadow-md
              ${getTextColor(stock)}
            `}
            style={{ backgroundColor: getHeatmapColor(stock) }}
            onClick={() => onStockSelect && onStockSelect(stock)}
            title={`${stock.symbol}: ${
              stock.price_change_pct > 0 ? '+' : ''
            }${stock.price_change_pct?.toFixed(
              1
            )}% | Vol: ${stock.volume_ratio?.toFixed(1)}x`}
          >
            {/* Type Icon */}
            <div className="absolute top-1 right-1">
              {getTypeIcon(stock.type)}
            </div>

            {/* Symbol */}
            <div className="font-bold text-sm mb-1 truncate pr-4">
              {stock.symbol}
            </div>

            {/* Price Change */}
            <div className="text-xs font-medium mb-1">
              {stock.price_change_pct > 0 ? '+' : ''}
              {stock.price_change_pct?.toFixed(1)}%
            </div>

            {/* Volume Ratio */}
            <div className="text-xs opacity-80">
              {stock.volume_ratio?.toFixed(1)}x vol
            </div>

            {/* Price */}
            <div className="text-xs mt-1 opacity-70">
              ${stock.close_price?.toFixed(2)}
            </div>

            {/* Intensity Indicator */}
            <div className="absolute bottom-1 left-1 w-2 h-1 bg-white opacity-30 rounded-full"></div>
          </div>
        ))}
      </div>

      {/* Summary Stats */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
        <div className="bg-green-50 rounded-lg p-3">
          <div className="text-lg font-bold text-green-600">
            {heatmapData.filter((s) => s.type === 'bullish').length}
          </div>
          <div className="text-sm text-green-700">Bullish</div>
        </div>
        <div className="bg-red-50 rounded-lg p-3">
          <div className="text-lg font-bold text-red-600">
            {heatmapData.filter((s) => s.type === 'bearish').length}
          </div>
          <div className="text-sm text-red-700">Bearish</div>
        </div>
        <div className="bg-blue-50 rounded-lg p-3">
          <div className="text-lg font-bold text-blue-600">
            {heatmapData.filter((s) => s.type === 'near_bullish').length}
          </div>
          <div className="text-sm text-blue-700">Near Bull</div>
        </div>
        <div className="bg-yellow-50 rounded-lg p-3">
          <div className="text-lg font-bold text-yellow-600">
            {heatmapData.filter((s) => s.type === 'near_bearish').length}
          </div>
          <div className="text-sm text-yellow-700">Near Bear</div>
        </div>
      </div>

      {/* Top Performers */}
      <div className="mt-6 bg-gray-50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Top Movers</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Biggest Gainer:</span>
            {(() => {
              const topGainer = heatmapData.reduce(
                (max, stock) =>
                  stock.price_change_pct > (max?.price_change_pct || -Infinity)
                    ? stock
                    : max,
                null
              );
              return topGainer ? (
                <div className="font-medium text-green-600">
                  {topGainer.symbol} (+{topGainer.price_change_pct?.toFixed(1)}
                  %)
                </div>
              ) : (
                <div className="text-gray-500">N/A</div>
              );
            })()}
          </div>
          <div>
            <span className="text-gray-600">Biggest Loser:</span>
            {(() => {
              const topLoser = heatmapData.reduce(
                (min, stock) =>
                  stock.price_change_pct < (min?.price_change_pct || Infinity)
                    ? stock
                    : min,
                null
              );
              return topLoser ? (
                <div className="font-medium text-red-600">
                  {topLoser.symbol} ({topLoser.price_change_pct?.toFixed(1)}%)
                </div>
              ) : (
                <div className="text-gray-500">N/A</div>
              );
            })()}
          </div>
          <div>
            <span className="text-gray-600">Highest Volume:</span>
            {(() => {
              const topVolume = heatmapData.reduce(
                (max, stock) =>
                  stock.volume_ratio > (max?.volume_ratio || 0) ? stock : max,
                null
              );
              return topVolume ? (
                <div className="font-medium text-blue-600">
                  {topVolume.symbol} ({topVolume.volume_ratio?.toFixed(1)}x)
                </div>
              ) : (
                <div className="text-gray-500">N/A</div>
              );
            })()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketHeatmap;
