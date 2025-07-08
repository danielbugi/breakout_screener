import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ExternalLink } from 'lucide-react';

const BreakoutTable = ({ title, data, type, icon: Icon, onViewMore }) => {
  const navigate = useNavigate();

  const handleRedirect = () => {
    console.log('Redirecting to breakouts page');
    navigate('/breakouts');
  };

  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center gap-2 mb-4">
          <Icon className="h-5 w-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
        <p className="text-gray-500 text-center py-8">
          No {type} breakouts found
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Icon className="h-5 w-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded">
            {data.length}
          </span>
        </div>
        {data.length > 5 && (
          <button
            onClick={() => handleRedirect()}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center gap-1"
          >
            View All <ExternalLink className="h-3 w-3" />
          </button>
        )}
      </div>

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
              {type === 'near' && (
                <th className="text-right py-2 font-medium text-gray-600">
                  Position
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 5).map((stock, index) => (
              <tr
                key={stock.symbol || index}
                className="border-b border-gray-100 hover:bg-gray-50"
              >
                <td className="py-3 font-medium text-gray-900">
                  {stock.symbol}
                </td>
                <td className="text-right py-3">
                  ${stock.close_price?.toFixed(2)}
                </td>
                <td
                  className={`text-right py-3 font-medium ${
                    stock.price_change_pct > 0
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}
                >
                  {stock.price_change_pct > 0 ? '+' : ''}
                  {stock.price_change_pct?.toFixed(1)}%
                </td>
                <td className="text-right py-3">
                  {stock.volume_ratio?.toFixed(1)}x
                </td>
                {type === 'near' && (
                  <td className="text-right py-3">
                    {stock.price_position_pct?.toFixed(1)}%
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default BreakoutTable;
