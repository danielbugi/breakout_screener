function ErrorMessage() {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
      <div className="flex items-center justify-center text-red-600 mb-2">
        <WifiOff className="h-5 w-5 mr-2" />
        <span className="font-medium">Connection Error</span>
      </div>
      <p className="text-red-700 mb-3">{message}</p>
      <button
        onClick={onRetry}
        className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
      >
        Try Again
      </button>
    </div>
  );
}
export default ErrorMessage;
