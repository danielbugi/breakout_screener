import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Search,
  Filter,
  Download,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  RefreshCw,
  Eye,
  Activity,
  BarChart3,
  X,
} from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';
import StockChart from './StockChart';
import apiService from '../services/apiService';

const BreakoutDetails = () => {
  // Data state
  const [scanData, setScanData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Filter and search state
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState('all');
  const [minVolumeRatio, setMinVolumeRatio] = useState('');
  const [minPriceChange, setMinPriceChange] = useState('');
  const [sortField, setSortField] = useState('symbol');
  const [sortDirection, setSortDirection] = useState('asc');

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(25);

  // Modal state for detailed view
  const [selectedStock, setSelectedStock] = useState(null);
  const [showChart, setShowChart] = useState(false);

  // Fetch data
  const fetchData = useCallback(async () => {
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
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Combine all breakouts into a single array with type labels
  const allBreakouts = useMemo(() => {
    if (!scanData) return [];

    const breakouts = [];

    // Add bullish breakouts
    scanData.breakouts.bullish?.forEach((stock) => {
      breakouts.push({
        ...stock,
        breakout_type: 'bullish',
        type_label: 'Bullish Breakout',
      });
    });

    // Add bearish breakouts
    scanData.breakouts.bearish?.forEach((stock) => {
      breakouts.push({
        ...stock,
        breakout_type: 'bearish',
        type_label: 'Bearish Breakout',
      });
    });

    // Add near bullish breakouts
    scanData.breakouts.near_bullish?.forEach((stock) => {
      breakouts.push({
        ...stock,
        breakout_type: 'near_bullish',
        type_label: 'Near Bullish',
      });
    });

    // Add near bearish breakouts
    scanData.breakouts.near_bearish?.forEach((stock) => {
      breakouts.push({
        ...stock,
        breakout_type: 'near_bearish',
        type_label: 'Near Bearish',
      });
    });

    return breakouts;
  }, [scanData]);

  // Filter and search logic
  const filteredBreakouts = useMemo(() => {
    let filtered = allBreakouts;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter((stock) =>
        stock.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Type filter
    if (selectedType !== 'all') {
      filtered = filtered.filter(
        (stock) => stock.breakout_type === selectedType
      );
    }

    // Volume ratio filter
    if (minVolumeRatio) {
      filtered = filtered.filter(
        (stock) => stock.volume_ratio >= parseFloat(minVolumeRatio)
      );
    }

    // Price change filter
    if (minPriceChange) {
      filtered = filtered.filter(
        (stock) =>
          Math.abs(stock.price_change_pct) >= parseFloat(minPriceChange)
      );
    }

    return filtered;
  }, [allBreakouts, searchTerm, selectedType, minVolumeRatio, minPriceChange]);

  // Sorting logic
  const sortedBreakouts = useMemo(() => {
    const sorted = [...filteredBreakouts].sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      // Handle different data types
      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return sorted;
  }, [filteredBreakouts, sortField, sortDirection]);

  // Pagination logic
  const paginatedBreakouts = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return sortedBreakouts.slice(startIndex, startIndex + itemsPerPage);
  }, [sortedBreakouts, currentPage, itemsPerPage]);

  const totalPages = Math.ceil(sortedBreakouts.length / itemsPerPage);

  // Sorting handler
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
    setCurrentPage(1); // Reset to first page when sorting
  };

  // Get sort icon
  const getSortIcon = (field) => {
    if (sortField !== field) return <ArrowUpDown className="h-4 w-4" />;
    return sortDirection === 'asc' ? (
      <ArrowUp className="h-4 w-4" />
    ) : (
      <ArrowDown className="h-4 w-4" />
    );
  };

  // Export to CSV
  const exportToCSV = () => {
    if (sortedBreakouts.length === 0) return;

    const headers = [
      'Symbol',
      'Type',
      'Price',
      'Change %',
      'Volume Ratio',
      'ATR %',
      'Position %',
      'Channel High',
      'Channel Low',
      'Days Since High',
      'Days Since Low',
      'Date',
    ];

    const csvContent = [
      headers.join(','),
      ...sortedBreakouts.map((stock) =>
        [
          stock.symbol,
          stock.type_label,
          stock.close_price,
          stock.price_change_pct,
          stock.volume_ratio,
          stock.atr_pct || '',
          stock.price_position_pct || '',
          stock.donchian_high,
          stock.donchian_low,
          stock.days_since_high || '',
          stock.days_since_low || '',
          stock.date,
        ].join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `donchian_breakouts_${
      new Date().toISOString().split('T')[0]
    }.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Get breakout type badge styling
  const getTypeBadge = (breakoutType) => {
    const styles = {
      bullish: 'bg-green-100 text-green-800',
      bearish: 'bg-red-100 text-red-800',
      near_bullish: 'bg-blue-100 text-blue-800',
      near_bearish: 'bg-yellow-100 text-yellow-800',
    };
    return styles[breakoutType] || 'bg-gray-100 text-gray-800';
  };

  // Handle stock selection for detailed view
  const handleStockSelect = (stock, withChart = false) => {
    setSelectedStock(stock);
    setShowChart(withChart);
  };

  if (loading) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Breakout Analysis
        </h1>
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Breakout Analysis
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

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Breakout Analysis
          </h1>
          <p className="text-gray-600 mt-1">
            Detailed analysis of {allBreakouts.length} breakout signals
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

      {/* Filters and Search */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Filter className="h-5 w-5" />
          Filters & Search
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="h-4 w-4 absolute left-3 top-3 text-gray-400" />
            <input
              type="text"
              placeholder="Search symbols..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
            />
          </div>

          {/* Type Filter */}
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          >
            <option value="all">All Types</option>
            <option value="bullish">Bullish Breakouts</option>
            <option value="bearish">Bearish Breakouts</option>
            <option value="near_bullish">Near Bullish</option>
            <option value="near_bearish">Near Bearish</option>
          </select>

          {/* Volume Ratio Filter */}
          <input
            type="number"
            placeholder="Min Volume Ratio"
            value={minVolumeRatio}
            onChange={(e) => setMinVolumeRatio(e.target.value)}
            step="0.1"
            min="0"
            className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />

          {/* Price Change Filter */}
          <input
            type="number"
            placeholder="Min Price Change %"
            value={minPriceChange}
            onChange={(e) => setMinPriceChange(e.target.value)}
            step="0.1"
            min="0"
            className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />

          {/* Export Button */}
          <button
            onClick={exportToCSV}
            disabled={sortedBreakouts.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 text-sm"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
        </div>

        {/* Results Summary */}
        <div className="mt-4 text-sm text-gray-600">
          Showing {sortedBreakouts.length} of {allBreakouts.length} breakouts
          {searchTerm && ` matching "${searchTerm}"`}
          {selectedType !== 'all' && ` of type "${selectedType}"`}
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                {[
                  { field: 'symbol', label: 'Symbol' },
                  { field: 'type_label', label: 'Type' },
                  { field: 'close_price', label: 'Price' },
                  { field: 'price_change_pct', label: 'Change %' },
                  { field: 'volume_ratio', label: 'Volume Ratio' },
                  { field: 'atr_pct', label: 'ATR %' },
                  { field: 'price_position_pct', label: 'Position %' },
                  { field: 'date', label: 'Date' },
                ].map(({ field, label }) => (
                  <th
                    key={field}
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort(field)}
                  >
                    <div className="flex items-center gap-1">
                      {label}
                      {getSortIcon(field)}
                    </div>
                  </th>
                ))}
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {paginatedBreakouts.map((stock, index) => (
                <tr
                  key={`${stock.symbol}-${index}`}
                  className="hover:bg-gray-50"
                >
                  <td className="px-4 py-3 font-medium text-gray-900">
                    {stock.symbol}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getTypeBadge(
                        stock.breakout_type
                      )}`}
                    >
                      {stock.type_label}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    ${stock.close_price?.toFixed(2)}
                  </td>
                  <td
                    className={`px-4 py-3 font-medium ${
                      stock.price_change_pct > 0
                        ? 'text-green-600'
                        : 'text-red-600'
                    }`}
                  >
                    {stock.price_change_pct > 0 ? '+' : ''}
                    {stock.price_change_pct?.toFixed(1)}%
                  </td>
                  <td className="px-4 py-3">
                    {stock.volume_ratio?.toFixed(1)}x
                  </td>
                  <td className="px-4 py-3">{stock.atr_pct?.toFixed(1)}%</td>
                  <td className="px-4 py-3">
                    {stock.price_position_pct?.toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {stock.date}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleStockSelect(stock, false)}
                        className="text-blue-600 hover:text-blue-700 p-1"
                        title="View Details"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => handleStockSelect(stock, true)}
                        className="text-green-600 hover:text-green-700 p-1"
                        title="View Chart"
                      >
                        <BarChart3 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="bg-gray-50 px-4 py-3 border-t border-gray-200 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              Show
              <select
                value={itemsPerPage}
                onChange={(e) => {
                  setItemsPerPage(Number(e.target.value));
                  setCurrentPage(1);
                }}
                className="border border-gray-300 rounded px-2 py-1 text-sm"
              >
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
              per page
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
              >
                Previous
              </button>

              <span className="text-sm text-gray-600">
                Page {currentPage} of {totalPages}
              </span>

              <button
                onClick={() =>
                  setCurrentPage(Math.min(totalPages, currentPage + 1))
                }
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Stock Detail Modal */}
      {selectedStock && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div
            className={`bg-white rounded-lg max-h-[90vh] overflow-y-auto ${
              showChart ? 'max-w-6xl w-full' : 'max-w-2xl w-full'
            }`}
          >
            {showChart ? (
              // Chart View
              <div className="p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-gray-900">
                    Chart Analysis - {selectedStock.symbol}
                  </h3>
                  <button
                    onClick={() => setSelectedStock(null)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X className="h-6 w-6" />
                  </button>
                </div>
                <StockChart
                  symbol={selectedStock.symbol}
                  stockData={selectedStock}
                />
              </div>
            ) : (
              // Detailed Data View
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-gray-900">
                    {selectedStock.symbol} - Detailed Analysis
                  </h3>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setShowChart(true)}
                      className="flex items-center gap-2 px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                      <BarChart3 className="h-4 w-4" />
                      View Chart
                    </button>
                    <button
                      onClick={() => setSelectedStock(null)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <X className="h-6 w-6" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <strong>Type:</strong> {selectedStock.type_label}
                  </div>
                  <div>
                    <strong>Price:</strong> $
                    {selectedStock.close_price?.toFixed(2)}
                  </div>
                  <div>
                    <strong>Price Change:</strong>
                    <span
                      className={
                        selectedStock.price_change_pct > 0
                          ? 'text-green-600'
                          : 'text-red-600'
                      }
                    >
                      {selectedStock.price_change_pct > 0 ? '+' : ''}
                      {selectedStock.price_change_pct?.toFixed(2)}%
                    </span>
                  </div>
                  <div>
                    <strong>Volume Ratio:</strong>{' '}
                    {selectedStock.volume_ratio?.toFixed(2)}x
                  </div>
                  <div>
                    <strong>ATR:</strong> {selectedStock.atr_pct?.toFixed(2)}%
                  </div>
                  <div>
                    <strong>Position:</strong>{' '}
                    {selectedStock.price_position_pct?.toFixed(1)}%
                  </div>
                  <div>
                    <strong>Channel High:</strong> $
                    {selectedStock.donchian_high?.toFixed(2)}
                  </div>
                  <div>
                    <strong>Channel Low:</strong> $
                    {selectedStock.donchian_low?.toFixed(2)}
                  </div>
                  <div>
                    <strong>Channel Mid:</strong> $
                    {selectedStock.donchian_mid?.toFixed(2)}
                  </div>
                  <div>
                    <strong>Channel Width:</strong>{' '}
                    {selectedStock.channel_width_pct?.toFixed(2)}%
                  </div>
                  <div>
                    <strong>Days Since High:</strong>{' '}
                    {selectedStock.days_since_high}
                  </div>
                  <div>
                    <strong>Days Since Low:</strong>{' '}
                    {selectedStock.days_since_low}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default BreakoutDetails;
