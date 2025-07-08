// Enhanced API service with proper health check handling
const API_BASE_URL = '/api';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
    this.timeout = 10000; // 10 seconds timeout
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      timeout: this.timeout,
      ...options,
    };

    // Create AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    config.signal = controller.signal;

    try {
      console.log(`🌐 API Request: ${endpoint}`);
      const response = await fetch(url, config);
      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorMessage;
        try {
          const errorData = await response.json();
          errorMessage =
            errorData.message ||
            `HTTP ${response.status}: ${response.statusText}`;
        } catch (parseError) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log(`✅ API Response (${endpoint}):`, data);
      return data;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        throw new Error(
          'Request timeout. Please check your connection and try again.'
        );
      }

      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error(
          'Cannot connect to server. Please ensure the backend is running on port 5000.'
        );
      }

      console.error(`❌ API Error (${endpoint}):`, error);
      throw error;
    }
  }

  // Fixed health check - works with actual Flask response
  async healthCheck() {
    try {
      const response = await this.request('/health');
      console.log('🏥 Health check raw response:', response);

      // Flask backend returns: { status: "healthy", timestamp: "...", ... }
      // Consider it connected if we get any successful response
      const isConnected =
        response && (response.status === 'healthy' || response.status);

      return {
        connected: isConnected,
        data: response,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error('🏥 Health check failed:', error);
      return {
        connected: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Get latest scan data
  async getLatestScan() {
    return this.request('/latest-scan');
  }

  // Get scan summary
  async getScanSummary() {
    return this.request('/scan-summary');
  }

  // Get scan history with error handling
  async getScanHistory(days = 7) {
    if (days < 1 || days > 30) {
      throw new Error('Days parameter must be between 1 and 30');
    }
    return this.request(`/scan-history?days=${days}`);
  }

  // Get breakouts by type with validation
  async getBreakoutsByType(type, page = 1, perPage = 20) {
    const validTypes = ['bullish', 'bearish', 'near_bullish', 'near_bearish'];
    if (!validTypes.includes(type)) {
      throw new Error(
        `Invalid breakout type. Must be one of: ${validTypes.join(', ')}`
      );
    }

    if (page < 1) page = 1;
    if (perPage < 1 || perPage > 100) perPage = 20;

    return this.request(`/breakouts/${type}?page=${page}&per_page=${perPage}`);
  }

  // Trigger new scan
  async triggerScan() {
    return this.request('/trigger-scan', {
      method: 'POST',
    });
  }

  // Test connection method with detailed debugging
  async testConnection() {
    try {
      const start = Date.now();
      const result = await this.healthCheck();
      const duration = Date.now() - start;

      console.log('🔗 Connection test result:', result);

      return {
        success: result.connected,
        duration,
        status: result.connected ? 'Connected' : 'Disconnected',
        error: result.error || null,
        details: result.data,
      };
    } catch (error) {
      console.error('🔗 Connection test failed:', error);
      return {
        success: false,
        duration: null,
        status: 'Connection Failed',
        error: error.message,
      };
    }
  }

  // Simple connectivity check
  async isHealthy() {
    try {
      const health = await this.healthCheck();
      return health.connected;
    } catch {
      return false;
    }
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService;
