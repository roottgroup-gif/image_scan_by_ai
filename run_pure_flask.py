#!/usr/bin/env python3
"""
Pure Flask Application Runner
Entry point for the AI Detection Checker web application (Python only)
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pure_flask_app import app

if __name__ == '__main__':
    print("🚀 Starting AI Detection Web Application (Pure Python Flask)")
    print("📁 Using local storage only (no database required)")
    print("🎨 All original design elements preserved")
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🌐 Server starting on http://0.0.0.0:{port}")
    
    # Run the Flask application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Disable debug to avoid reloader issues in Replit
        use_reloader=False,
        threaded=True  # Enable threading for better performance
    )