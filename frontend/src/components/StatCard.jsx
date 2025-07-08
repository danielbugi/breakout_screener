import React from 'react';

const StatCard = ({ title, value, change, icon: Icon, color, bgColor }) => (
  <div className={`p-6 rounded-xl border border-gray-200 ${bgColor}`}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
        <p className={`text-2xl font-bold ${color}`}>{value}</p>
        {change && <p className="text-sm text-gray-500 mt-1">{change}</p>}
      </div>
      <Icon className={`h-6 w-6 ${color}`} />
    </div>
  </div>
);

export default StatCard;