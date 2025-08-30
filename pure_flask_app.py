#!/usr/bin/env python3
"""
AI Image Detection Web Application - Pure Python Flask
Complete conversion from Node.js/React to Python Flask
Maintains all original design elements while using local storage only
"""

import os
import sys
import json
import time
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np

# Try to import AI detector
try:
    from simple_ai_detector import SimpleAIImageDetector
    detector = SimpleAIImageDetector()
    print("AI detector loaded successfully", file=sys.stderr)
except Exception as e:
    print(f"Error loading AI detector: {e}", file=sys.stderr)
    detector = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_buffer(image_buffer):
    """Process image buffer to get metadata"""
    try:
        pil_image = Image.open(BytesIO(image_buffer))
        width, height = pil_image.size
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        file_size = len(image_buffer)
        
        return {
            'width': width,
            'height': height,
            'size': file_size,
            'format': pil_image.format or 'Unknown'
        }
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        return None

@app.route('/')
def home():
    """Home page with AI detection interface"""
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    """How it works page"""
    return render_template('how-it-works.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/faq')
def faq():
    """FAQ page"""
    return render_template('faq.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for image analysis"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: PNG, JPG, JPEG, GIF, BMP, WebP'}), 400
        
        # Read image data
        image_buffer = file.read()
        
        # Process image metadata
        metadata = process_image_buffer(image_buffer)
        if not metadata:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Perform AI detection analysis
        start_time = time.time()
        
        if detector:
            try:
                analysis_result = detector.detect_ai_image(image_buffer)
            except Exception as e:
                print(f"Analysis error: {e}", file=sys.stderr)
                analysis_result = {
                    'classification': 'Analysis Error',
                    'confidence': 50,
                    'indicators': ['Technical analysis failed', 'Manual review required'],
                    'method_used': 'Error Recovery'
                }
        else:
            analysis_result = {
                'classification': 'Detector Unavailable',
                'confidence': 50,
                'indicators': ['AI detector not loaded', 'Manual review required'],
                'method_used': 'Error Recovery'
            }
        
        processing_time = round(time.time() - start_time, 2)
        
        # Prepare response
        response_data = {
            'classification': analysis_result.get('classification', 'Unknown'),
            'confidence': analysis_result.get('confidence', 50),
            'processingTime': processing_time,
            'imageSize': f"{metadata['width']}x{metadata['height']}",
            'indicators': [
                {'name': indicator, 'strength': 'Strong' if i < 2 else 'Moderate' if i < 4 else 'Weak'} 
                for i, indicator in enumerate(analysis_result.get('indicators', [])[:6])
            ],
            'filename': secure_filename(file.filename or 'unknown.jpg'),
            'originalSize': metadata['size'],
            'methodUsed': analysis_result.get('method_used', 'Computer Vision Forensics')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"API error: {e}", file=sys.stderr)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('404.html'), 404

if __name__ == '__main__':
    print("🚀 Starting AI Detection Web Application (Pure Flask)", file=sys.stderr)
    print("📁 Using local storage only (no database required)", file=sys.stderr)
    print("🎨 All original design elements preserved", file=sys.stderr)
    
    # Get port from environment variable or default to 5000 (frontend port)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🌐 Server starting on http://0.0.0.0:{port}", file=sys.stderr)
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False,  # Disable debug to avoid reloader issues  
        use_reloader=False
    )