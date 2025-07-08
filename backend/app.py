#!/usr/bin/env python3
"""
Flask API Backend for Donchian Breakout Screener
Serves data from automated screening results
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import json
import glob
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
DATA_DIR = 'data'  # Directory where screener outputs are stored
AUTOMATION_DIR = '../automation'  # Path to automation files

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


class BreakoutDataAPI:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

    def get_latest_scan_file(self):
        """Get the most recent breakout scan file"""
        try:
            # Look for JSON files from the screener
            pattern = os.path.join(self.data_dir, 'donchian_breakouts_*.json')
            files = glob.glob(pattern)

            if not files:
                return None

            # Sort by modification time, get the latest
            latest_file = max(files, key=os.path.getmtime)
            return latest_file

        except Exception as e:
            logger.error(f"Error finding latest scan file: {e}")
            return None

    def load_scan_data(self, file_path):
        """Load and parse scan data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading scan data from {file_path}: {e}")
            return None

    def get_scan_history(self, days=7):
        """Get scan files from the last N days"""
        try:
            pattern = os.path.join(self.data_dir, 'donchian_breakouts_*.json')
            files = glob.glob(pattern)

            if not files:
                return []

            # Filter files from last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_files = []

            for file_path in files:
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time >= cutoff_date:
                    recent_files.append({
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path),
                        'scan_time': file_time.isoformat(),
                        'file_size': os.path.getsize(file_path)
                    })

            # Sort by scan time, newest first
            recent_files.sort(key=lambda x: x['scan_time'], reverse=True)
            return recent_files

        except Exception as e:
            logger.error(f"Error getting scan history: {e}")
            return []


# Initialize API instance
api = BreakoutDataAPI()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_dir': DATA_DIR,
        'automation_dir': AUTOMATION_DIR
    })


@app.route('/api/latest-scan', methods=['GET'])
def get_latest_scan():
    """Get the latest breakout scan data"""
    try:
        latest_file = api.get_latest_scan_file()

        if not latest_file:
            return jsonify({
                'success': False,
                'message': 'No scan data available',
                'error': 'NO_DATA_FOUND'
            }), 404

        scan_data = api.load_scan_data(latest_file)

        if not scan_data:
            return jsonify({
                'success': False,
                'message': 'Failed to load scan data',
                'error': 'DATA_LOAD_ERROR'
            }), 500

        # Add file metadata
        file_stats = os.stat(latest_file)
        scan_data['file_info'] = {
            'filename': os.path.basename(latest_file),
            'file_size': file_stats.st_size,
            'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        }

        return jsonify({
            'success': True,
            'data': scan_data,
            'cached': True
        })

    except Exception as e:
        logger.error(f"Error in get_latest_scan: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error',
            'error': str(e)
        }), 500


@app.route('/api/scan-history', methods=['GET'])
def get_scan_history():
    """Get historical scan data"""
    try:
        days = request.args.get('days', 7, type=int)
        days = min(days, 30)  # Limit to 30 days max

        history = api.get_scan_history(days)

        return jsonify({
            'success': True,
            'data': history,
            'days_requested': days,
            'total_scans': len(history)
        })

    except Exception as e:
        logger.error(f"Error in get_scan_history: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to get scan history',
            'error': str(e)
        }), 500


@app.route('/api/scan-summary', methods=['GET'])
def get_scan_summary():
    """Get summary statistics from the latest scan"""
    try:
        latest_file = api.get_latest_scan_file()

        if not latest_file:
            return jsonify({
                'success': False,
                'message': 'No scan data available'
            }), 404

        scan_data = api.load_scan_data(latest_file)

        if not scan_data or 'metadata' not in scan_data:
            return jsonify({
                'success': False,
                'message': 'Invalid scan data format'
            }), 500

        metadata = scan_data['metadata']
        breakouts = scan_data.get('breakouts', {})

        summary = {
            'scan_date': metadata.get('scan_date'),
            'period': metadata.get('period'),
            'total_scanned': metadata.get('total_scanned', 0),
            'breakout_counts': {
                'bullish': metadata.get('bullish_count', 0),
                'bearish': metadata.get('bearish_count', 0),
                'near_bullish': len(breakouts.get('near_bullish', [])),
                'near_bearish': len(breakouts.get('near_bearish', [])),
                'total_breakouts': metadata.get('bullish_count', 0) + metadata.get('bearish_count', 0)
            },
            'top_performers': scan_data.get('summary_stats', {}).get('top_performers', [])[:5],
            'scan_efficiency': {
                'success_rate': ((metadata.get('total_scanned', 0) - metadata.get('failed_count', 0)) / max(
                    metadata.get('total_scanned', 1), 1)) * 100,
                'failed_count': metadata.get('failed_count', 0)
            }
        }

        return jsonify({
            'success': True,
            'data': summary
        })

    except Exception as e:
        logger.error(f"Error in get_scan_summary: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to get scan summary',
            'error': str(e)
        }), 500


@app.route('/api/breakouts/<breakout_type>', methods=['GET'])
def get_breakouts_by_type(breakout_type):
    """Get breakouts filtered by type (bullish, bearish, near_bullish, near_bearish)"""
    try:
        valid_types = ['bullish', 'bearish', 'near_bullish', 'near_bearish']

        if breakout_type not in valid_types:
            return jsonify({
                'success': False,
                'message': f'Invalid breakout type. Valid types: {valid_types}'
            }), 400

        latest_file = api.get_latest_scan_file()

        if not latest_file:
            return jsonify({
                'success': False,
                'message': 'No scan data available'
            }), 404

        scan_data = api.load_scan_data(latest_file)

        if not scan_data:
            return jsonify({
                'success': False,
                'message': 'Failed to load scan data'
            }), 500

        breakouts = scan_data.get('breakouts', {}).get(breakout_type, [])

        # Add pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 100)  # Max 100 items per page

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_breakouts = breakouts[start_idx:end_idx]

        return jsonify({
            'success': True,
            'data': paginated_breakouts,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_items': len(breakouts),
                'total_pages': (len(breakouts) + per_page - 1) // per_page
            },
            'breakout_type': breakout_type
        })

    except Exception as e:
        logger.error(f"Error in get_breakouts_by_type: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to get breakout data',
            'error': str(e)
        }), 500


@app.route('/api/trigger-scan', methods=['POST'])
def trigger_new_scan():
    """Trigger a new scan (for development/testing)"""
    try:
        # This would typically trigger the automation script
        # For now, just return a message
        return jsonify({
            'success': True,
            'message': 'Scan trigger received. Note: Implement automation integration.',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in trigger_new_scan: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to trigger scan',
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': 'Endpoint not found',
        'error': 'NOT_FOUND'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': 'Internal server error',
        'error': 'INTERNAL_ERROR'
    }), 500


if __name__ == '__main__':
    print("🚀 Starting Donchian Breakout API Server...")
    print(f"📁 Data directory: {os.path.abspath(DATA_DIR)}")
    print(f"🔗 API will be available at: http://localhost:5000/api/")
    print("\n📋 Available endpoints:")
    print("   GET  /api/health              - Health check")
    print("   GET  /api/latest-scan         - Latest scan data")
    print("   GET  /api/scan-summary        - Scan summary")
    print("   GET  /api/scan-history        - Historical scans")
    print("   GET  /api/breakouts/<type>    - Breakouts by type")
    print("   POST /api/trigger-scan        - Trigger new scan")
    print("\n✅ Server starting...")

    app.run(debug=True, host='0.0.0.0', port=5000)