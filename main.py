#!/usr/bin/env python3
"""
AI Image Detection Web Application - Main Entry Point
Pure Python Flask implementation with all original design elements preserved
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    print("🚀 Starting AI Detection Web Application")
    print("📁 Pure Python Flask implementation")
    print("🎨 All original design elements preserved")
    
    # Import and run the Flask application
    from pure_flask_app import app
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🌐 Server starting on http://0.0.0.0:{port}")
    
    # Run the Flask application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Disable debug to avoid reloader issues
        use_reloader=False,
        threaded=True
    )