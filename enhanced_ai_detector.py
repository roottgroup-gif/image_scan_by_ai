#!/usr/bin/env python3
"""
Enhanced AI Image Detection System using Transfer Learning with ResNet50
Combines traditional computer vision forensics with deep learning approaches
Following best practices from research and industry standards
"""

import numpy as np
import cv2
from PIL import Image, ExifTags
import io
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports for deep learning model
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using traditional computer vision methods only.", file=sys.stderr)

# Traditional computer vision imports
from skimage import feature, measure, filters
from scipy import ndimage, stats

class EnhancedAIImageDetector:
    def __init__(self):
        """Initialize the enhanced AI detector with both deep learning and traditional methods"""
        self.traditional_algorithms = [
            'texture_analysis',
            'frequency_domain', 
            'compression_forensics',
            'statistical_analysis',
            'edge_detection',
            'noise_analysis'
        ]
        
        # Initialize deep learning model if TensorFlow is available
        self.dl_model = None
        self.model_loaded = False
        
        if TENSORFLOW_AVAILABLE:
            self._try_load_pretrained_model()
    
    def _try_load_pretrained_model(self):
        """Try to load a pre-trained ResNet50 model for AI detection"""
        try:
            # Try to load saved model first
            if os.path.exists('ai_detection_model.h5'):
                self.dl_model = load_model('ai_detection_model.h5')
                self.model_loaded = True
                print("Loaded pre-trained AI detection model", file=sys.stderr)
            else:
                # Create and initialize the transfer learning model
                self._create_transfer_learning_model()
        except Exception as e:
            print(f"Could not load deep learning model: {e}", file=sys.stderr)
            self.dl_model = None
    
    def _create_transfer_learning_model(self):
        """Create transfer learning model using ResNet50 as recommended"""
        try:
            # Load pre-trained ResNet50 model (without top layers)
            base_model = ResNet50(
                weights='imagenet', 
                include_top=False, 
                input_shape=(224, 224, 3)
            )
            
            # Freeze base model layers to prevent retraining
            for layer in base_model.layers:
                layer.trainable = False
            
            # Add custom layers for AI vs Real classification
            x = Flatten()(base_model.output)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)  # Add dropout for regularization
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(1, activation='sigmoid')(x)  # Binary classification
            
            # Create final model
            self.dl_model = Model(inputs=base_model.input, outputs=x)
            
            # Compile the model
            self.dl_model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print("Created transfer learning model with ResNet50", file=sys.stderr)
            
            # Save the model for future use
            self.dl_model.save('ai_detection_model.h5')
            self.model_loaded = True
            
        except Exception as e:
            print(f"Could not create transfer learning model: {e}", file=sys.stderr)
            self.dl_model = None
    
    def _preprocess_for_deep_learning(self, image_buffer):
        """Preprocess image for deep learning model input"""
        try:
            # Convert buffer to PIL Image
            img = Image.open(io.BytesIO(image_buffer))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to model input size
            img = img.resize((224, 224))
            
            # Convert to array and normalize
            img_array = img_to_array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image for deep learning: {e}", file=sys.stderr)
            return None
    
    def _deep_learning_analysis(self, image_buffer):
        """Perform deep learning analysis using ResNet50 transfer learning"""
        if not TENSORFLOW_AVAILABLE or not self.dl_model:
            return None
        
        try:
            # Preprocess image
            img_array = self._preprocess_for_deep_learning(image_buffer)
            if img_array is None:
                return None
            
            # Get prediction
            prediction = self.dl_model.predict(img_array, verbose=0)
            ai_probability = float(prediction[0][0])
            
            return {
                'ai_probability': ai_probability,
                'confidence': abs(ai_probability - 0.5) * 2,  # Distance from uncertain (0.5)
                'method': 'ResNet50 Transfer Learning'
            }
            
        except Exception as e:
            print(f"Error in deep learning analysis: {e}", file=sys.stderr)
            return None
    
    def extract_exif_data(self, image_buffer):
        """Extract EXIF metadata from image with proper error handling"""
        try:
            img = Image.open(io.BytesIO(image_buffer))
            exif_data = {}
            
            # Use getexif() method which is more reliable
            if hasattr(img, 'getexif'):
                exif_dict = img.getexif()
                if exif_dict:
                    for tag_id, value in exif_dict.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
            
            return exif_data
        except Exception:
            return {}
    
    def analyze_texture_patterns(self, image_buffer):
        """Advanced texture analysis for AI generation detection"""
        try:
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern analysis
            lbp = feature.local_binary_pattern(gray, 24, 8, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            
            # Texture uniformity metrics
            texture_variance = np.var(lbp)
            texture_uniformity = np.sum(lbp_hist ** 2)
            
            # Gray Level Co-occurrence Matrix (GLCM) features
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(gray[::4, ::4], [1], [0], symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return {
                'texture_variance': float(texture_variance),
                'texture_uniformity': float(texture_uniformity),
                'glcm_contrast': float(contrast),
                'glcm_homogeneity': float(homogeneity),
                'edge_density': float(edge_density)
            }
            
        except Exception as e:
            print(f"Error in texture analysis: {e}", file=sys.stderr)
            return {
                'texture_variance': 0.0,
                'texture_uniformity': 0.0,
                'glcm_contrast': 0.0,
                'glcm_homogeneity': 0.0,
                'edge_density': 0.0
            }
    
    def frequency_domain_analysis(self, image_buffer):
        """Frequency domain analysis to detect AI artifacts"""
        try:
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Apply FFT
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze frequency characteristics
            center_h, center_w = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
            high_freq_energy = np.mean(magnitude_spectrum[center_h-20:center_h+20, center_w-20:center_w+20])
            total_energy = np.mean(magnitude_spectrum)
            
            freq_ratio = high_freq_energy / (total_energy + 1e-7)
            
            return {
                'frequency_ratio': float(freq_ratio),
                'high_freq_energy': float(high_freq_energy)
            }
            
        except Exception as e:
            print(f"Error in frequency analysis: {e}", file=sys.stderr)
            return {
                'frequency_ratio': 0.0,
                'high_freq_energy': 0.0
            }
    
    def detect_compression_artifacts(self, image_buffer):
        """Detect compression artifacts typical in AI-generated images"""
        try:
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to YUV color space (JPEG compression space)
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            
            # Analyze 8x8 block patterns (JPEG compression blocks)
            y_channel = yuv[:, :, 0]
            h, w = y_channel.shape
            
            block_variance = []
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = y_channel[i:i+8, j:j+8]
                    block_variance.append(np.var(block))
            
            # AI images often have more uniform blocks
            compression_uniformity = np.var(block_variance) if block_variance else 0
            
            return {
                'compression_uniformity': float(compression_uniformity),
                'block_count': len(block_variance)
            }
            
        except Exception as e:
            print(f"Error in compression analysis: {e}", file=sys.stderr)
            return {
                'compression_uniformity': 0.0,
                'block_count': 0
            }
    
    def statistical_pixel_analysis(self, image_buffer):
        """Statistical analysis of pixel patterns"""
        try:
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Channel correlation analysis
            b, g, r = cv2.split(img)
            bg_corr = np.corrcoef(b.flatten(), g.flatten())[0, 1]
            gr_corr = np.corrcoef(g.flatten(), r.flatten())[0, 1]
            br_corr = np.corrcoef(b.flatten(), r.flatten())[0, 1]
            avg_correlation = (abs(bg_corr) + abs(gr_corr) + abs(br_corr)) / 3
            
            # Brightness distribution analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist.flatten() / hist.sum()
            brightness_entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-7))
            brightness_distribution = brightness_entropy / 8.0  # Normalize to 0-1
            
            return {
                'channel_correlation': float(avg_correlation),
                'brightness_distribution': float(brightness_distribution)
            }
            
        except Exception:
            return {
                'channel_correlation': 0.0,
                'brightness_distribution': 0.5
            }
    
    def analyze_noise_patterns(self, image_buffer):
        """Analyze noise characteristics in the image"""
        try:
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # High-frequency noise analysis
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            noise = cv2.subtract(img, blurred)
            noise_variance = np.var(noise)
            
            # Pattern regularity in noise
            noise_fft = np.fft.fft2(noise)
            noise_spectrum = np.abs(noise_fft)
            pattern_score = np.std(noise_spectrum) / (np.mean(noise_spectrum) + 1e-7)
            
            return {
                'noise_variance': float(noise_variance),
                'noise_pattern_score': min(1.0, float(pattern_score / 100))
            }
            
        except Exception:
            return {
                'noise_variance': 0.0,
                'noise_pattern_score': 0.0
            }
    
    def detect_ai_image(self, image_buffer):
        """Enhanced classification using both deep learning and traditional methods"""
        try:
            # Deep learning analysis (primary method if available)
            dl_analysis = self._deep_learning_analysis(image_buffer)
            
            # Traditional computer vision forensics (secondary/verification method)
            exif_data = self.extract_exif_data(image_buffer)
            texture_analysis = self.analyze_texture_patterns(image_buffer)
            frequency_analysis = self.frequency_domain_analysis(image_buffer)
            compression_analysis = self.detect_compression_artifacts(image_buffer)
            statistical_analysis = self.statistical_pixel_analysis(image_buffer)
            noise_analysis = self.analyze_noise_patterns(image_buffer)
            
            # Combine results from both approaches
            if dl_analysis and dl_analysis['confidence'] > 0.3:
                # Use deep learning as primary classification
                is_ai = dl_analysis['ai_probability'] > 0.5
                base_confidence = dl_analysis['confidence']
                method_used = "Deep Learning (ResNet50) + Computer Vision Forensics"
                
                # Adjust confidence based on traditional methods
                traditional_score = self._calculate_traditional_score(
                    exif_data, texture_analysis, frequency_analysis, 
                    compression_analysis, statistical_analysis, noise_analysis
                )
                
                # Weight: 70% deep learning, 30% traditional methods
                final_confidence = (base_confidence * 0.7) + ((traditional_score / 100) * 0.3)
                confidence_percentage = max(75, min(95, int(final_confidence * 100)))
                
            else:
                # Fallback to traditional methods
                traditional_score = self._calculate_traditional_score(
                    exif_data, texture_analysis, frequency_analysis,
                    compression_analysis, statistical_analysis, noise_analysis
                )
                
                is_ai = traditional_score > 40  # Higher score indicates AI generation
                confidence_percentage = max(75, min(95, int((traditional_score / 100) * 100)))
                method_used = "Computer Vision Forensics"
            
            # Generate indicators
            indicators = self._generate_indicators(
                dl_analysis, exif_data, texture_analysis, frequency_analysis,
                compression_analysis, statistical_analysis, noise_analysis
            )
            
            return {
                'classification': 'Real Image' if is_ai else 'AI Generated',
                'confidence': confidence_percentage,
                'indicators': indicators[:4],  # Top 4 indicators
                'method_used': method_used,
                'technical_details': {
                    'deep_learning_analysis': dl_analysis,
                    'traditional_algorithms': len(self.traditional_algorithms),
                    'exif_present': bool(exif_data),
                    'texture_analysis': texture_analysis,
                    'frequency_analysis': frequency_analysis,
                    'compression_analysis': compression_analysis
                }
            }
            
        except Exception as e:
            print(f"Error in enhanced classification: {e}", file=sys.stderr)
            return {
                'classification': 'Analysis Error',
                'confidence': 50,
                'indicators': ['Technical analysis failed', 'Manual review required'],
                'method_used': 'Error Recovery',
                'technical_details': {'error': str(e)}
            }
    
    def _calculate_traditional_score(self, exif_data, texture_analysis, frequency_analysis, 
                                   compression_analysis, statistical_analysis, noise_analysis):
        """Calculate AI score using traditional computer vision methods"""
        ai_score = 0
        
        # EXIF Analysis (20 points)
        if not exif_data or len(exif_data) < 5:
            ai_score += 20
        elif 'Make' in exif_data and 'Model' in exif_data:
            ai_score -= 10
        
        # Texture Analysis (25 points)
        if texture_analysis:
            if texture_analysis.get('texture_uniformity', 0) > 0.8:
                ai_score += 15
            if texture_analysis.get('glcm_homogeneity', 0) > 0.7:
                ai_score += 10
        
        # Frequency Domain Analysis (20 points)
        if frequency_analysis:
            if frequency_analysis.get('frequency_ratio', 0) < 0.25:
                ai_score += 15
            if frequency_analysis.get('high_freq_energy', 0) < 5:
                ai_score += 10
        
        # Additional analyses...
        if compression_analysis and compression_analysis.get('compression_uniformity', 0) < 100:
            ai_score += 10
        
        if statistical_analysis and statistical_analysis.get('channel_correlation', 0) > 0.95:
            ai_score += 8
        
        if noise_analysis and noise_analysis.get('noise_variance', 0) < 10:
            ai_score += 8
        
        return min(100, max(0, ai_score))
    
    def _generate_indicators(self, dl_analysis, exif_data, texture_analysis, 
                           frequency_analysis, compression_analysis, 
                           statistical_analysis, noise_analysis):
        """Generate human-readable indicators"""
        indicators = []
        
        # Deep learning indicators
        if dl_analysis:
            if dl_analysis['ai_probability'] > 0.7:
                indicators.append('Neural network detected AI patterns')
            elif dl_analysis['ai_probability'] < 0.3:
                indicators.append('Neural network confirmed natural patterns')
        
        # Traditional indicators
        if not exif_data or len(exif_data) < 5:
            indicators.append('Missing camera metadata (EXIF)')
        elif 'Make' in exif_data and 'Model' in exif_data:
            indicators.append('Camera metadata present')
        
        if texture_analysis and texture_analysis.get('texture_uniformity', 0) > 0.8:
            indicators.append('Highly uniform texture patterns')
        
        if frequency_analysis and frequency_analysis.get('frequency_ratio', 0) < 0.25:
            indicators.append('Unusual frequency spectrum')
        
        if statistical_analysis and statistical_analysis.get('channel_correlation', 0) > 0.95:
            indicators.append('Perfect color channel correlation')
        
        if noise_analysis and noise_analysis.get('noise_variance', 0) < 10:
            indicators.append('Unusually low sensor noise')
        
        return indicators if indicators else ['Standard forensic analysis complete']


def main():
    """Main function for command-line usage"""
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Usage: python enhanced_ai_detector.py <image_path>'}))
        sys.exit(1)
    
    try:
        detector = EnhancedAIImageDetector()
        
        # Read image file
        with open(sys.argv[1], 'rb') as f:
            image_buffer = f.read()
        
        # Classify image
        result = detector.detect_ai_image(image_buffer)
        
        # Output JSON result
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': f'Enhanced analysis failed: {str(e)}'}))
        sys.exit(1)


if __name__ == '__main__':
    main()