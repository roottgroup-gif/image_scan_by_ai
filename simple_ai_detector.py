#!/usr/bin/env python3
"""
Simplified AI Image Detection System
Uses only traditional computer vision methods without TensorFlow dependency
Fallback for when TensorFlow is not available or has compatibility issues
"""

import numpy as np
import cv2
from PIL import Image, ExifTags
import io
import json
import sys
import os
from skimage import feature, measure, filters
from scipy import ndimage, stats
import warnings
warnings.filterwarnings('ignore')

class SimpleAIImageDetector:
    def __init__(self):
        """Initialize the simple AI detector with traditional computer vision only"""
        self.detection_algorithms = [
            'texture_analysis',
            'frequency_domain', 
            'compression_forensics',
            'statistical_analysis',
            'edge_detection',
            'noise_analysis'
        ]
    
    def extract_exif_data(self, image_buffer):
        """Extract EXIF metadata from image"""
        try:
            img = Image.open(io.BytesIO(image_buffer))
            exif_data = {}
            
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
                'texture_uniformity': float(texture_uniformity),
                'texture_variance': float(texture_variance),
                'glcm_contrast': float(contrast),
                'glcm_dissimilarity': float(dissimilarity),
                'glcm_homogeneity': float(homogeneity),
                'edge_density': float(edge_density)
            }
            
        except Exception as e:
            print(f"Error in texture analysis: {e}", file=sys.stderr)
            return {
                'texture_uniformity': 0.0,
                'texture_variance': 0.0,
                'glcm_contrast': 0.0,
                'glcm_dissimilarity': 0.0,
                'glcm_homogeneity': 0.0,
                'edge_density': 0.0
            }
    
    def frequency_domain_analysis(self, image_buffer):
        """Analyze frequency domain characteristics"""
        try:
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Fourier Transform
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
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
    
    def error_level_analysis(self, image_buffer):
        """Professional Error Level Analysis (ELA) for detecting manipulations"""
        try:
            # Load original image
            nparr = np.frombuffer(image_buffer, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Re-save at lower quality to detect compression inconsistencies
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, compressed_buffer = cv2.imencode('.jpg', img, encode_param)
            compressed_img = cv2.imdecode(compressed_buffer, cv2.IMREAD_COLOR)
            
            # Calculate differences
            diff = cv2.absdiff(img, compressed_img)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Analyze compression artifacts
            ela_variance = np.var(diff_gray)
            ela_mean = np.mean(diff_gray)
            ela_std = np.std(diff_gray)
            
            # Check for suspicious patterns
            suspicious_regions = np.sum(diff_gray > (ela_mean + 2 * ela_std))
            total_pixels = diff_gray.shape[0] * diff_gray.shape[1]
            suspicious_ratio = suspicious_regions / total_pixels
            
            return {
                'ela_variance': float(ela_variance),
                'ela_mean': float(ela_mean),
                'suspicious_ratio': float(suspicious_ratio)
            }
            
        except Exception as e:
            print(f"ELA analysis error: {e}", file=sys.stderr)
            return {
                'ela_variance': 0.0,
                'ela_mean': 0.0,
                'suspicious_ratio': 0.0
            }
    
    def detect_ai_image(self, image_buffer):
        """Simple classification using traditional computer vision methods"""
        try:
            # Professional forensic analysis using multiple techniques
            exif_data = self.extract_exif_data(image_buffer)
            texture_analysis = self.analyze_texture_patterns(image_buffer)
            frequency_analysis = self.frequency_domain_analysis(image_buffer)
            compression_analysis = self.detect_compression_artifacts(image_buffer)
            statistical_analysis = self.statistical_pixel_analysis(image_buffer)
            noise_analysis = self.analyze_noise_patterns(image_buffer)
            ela_analysis = self.error_level_analysis(image_buffer)
            
            # Calculate AI score using professional forensic methods
            traditional_score = self._calculate_traditional_score(
                exif_data, texture_analysis, frequency_analysis,
                compression_analysis, statistical_analysis, noise_analysis, ela_analysis
            )
            
            # Higher score means more AI-like characteristics detected
            is_ai = traditional_score > 45  # Higher score means AI generated
            
            # Standard confidence calculation
            if traditional_score >= 75:
                confidence_percentage = min(95, 85 + (traditional_score - 75) // 3)
            elif traditional_score >= 60:
                confidence_percentage = 80 + (traditional_score - 60) // 3
            elif traditional_score >= 45:
                confidence_percentage = 75 + (traditional_score - 45) // 5
            elif traditional_score >= 30:
                confidence_percentage = 65 + (30 - traditional_score) // 3  # Lower score = higher confidence for Real
            else:
                confidence_percentage = max(75, 80 - traditional_score // 2)  # Very low score = high confidence Real
            
            # Generate indicators
            indicators = self._generate_indicators(
                exif_data, texture_analysis, frequency_analysis,
                compression_analysis, statistical_analysis, noise_analysis
            )
            
            return {
                'classification': 'Real Image' if is_ai else 'AI Generated',
                'confidence': confidence_percentage,
                'indicators': indicators[:6],  # Top 6 indicators
                'method_used': 'Computer Vision Forensics',
                'technical_details': {
                    'traditional_algorithms': len(self.detection_algorithms),
                    'exif_present': bool(exif_data),
                    'texture_analysis': texture_analysis,
                    'frequency_analysis': frequency_analysis,
                    'compression_analysis': compression_analysis
                }
            }
            
        except Exception as e:
            print(f"Error in classification: {e}", file=sys.stderr)
            return {
                'classification': 'Analysis Error',
                'confidence': 50,
                'indicators': ['Technical analysis failed', 'Manual review required'],
                'method_used': 'Error Recovery',
                'technical_details': {'error': str(e)}
            }
    
    def _calculate_traditional_score(self, exif_data, texture_analysis, frequency_analysis, 
                                   compression_analysis, statistical_analysis, noise_analysis, ela_analysis=None):
        """Calculate AI score using professional forensic analysis techniques"""
        ai_score = 0
        total_checks = 0
        
        # EXIF Analysis (18 points) - Enhanced metadata evaluation for AI detection
        if not exif_data or len(exif_data) == 0:
            ai_score += 18  # Complete absence is highly suspicious for AI
        elif len(exif_data) < 3:
            ai_score += 12   # Very limited metadata indicates AI generation
        elif not ('Make' in exif_data and 'Model' in exif_data):
            ai_score += 6   # Missing camera info suggests AI generation
        elif 'DateTime' not in exif_data:
            ai_score += 3   # Missing timestamp is moderately suspicious
        total_checks += 18
        
        # Texture Analysis (22 points) - Enhanced pattern recognition for AI detection
        if texture_analysis:
            texture_uniformity = texture_analysis.get('texture_uniformity', 0)
            glcm_homogeneity = texture_analysis.get('glcm_homogeneity', 0)
            edge_density = texture_analysis.get('edge_density', 0)
            
            # More sensitive to AI texture patterns
            if texture_uniformity > 0.85:  # High uniformity indicates AI generation
                ai_score += 12
            elif texture_uniformity > 0.75:  # Moderate uniformity is suspicious
                ai_score += 8
            elif texture_uniformity > 0.65:  # Some uniformity suggests processing
                ai_score += 4
            
            if glcm_homogeneity > 0.75:  # Smooth textures typical of AI
                ai_score += 10
            elif glcm_homogeneity > 0.65:  # Moderately smooth
                ai_score += 6
                
            # More sensitive to edge anomalies
            if edge_density < 0.05:  # Very few edges (AI smoothing)
                ai_score += 8
            elif edge_density > 0.3:  # Excessive edges (AI artifacts)
                ai_score += 6
        total_checks += 22
        
        # Frequency Domain Analysis (20 points) - Enhanced spectral analysis
        if frequency_analysis:
            freq_ratio = frequency_analysis.get('frequency_ratio', 0)
            high_freq_energy = frequency_analysis.get('high_freq_energy', 0)
            
            # More sensitive to frequency anomalies typical of AI
            if freq_ratio < 0.15:  # Low high-frequency content (AI smoothing)
                ai_score += 12
            elif freq_ratio < 0.25:  # Reduced high-frequency content
                ai_score += 8
            elif freq_ratio < 0.35:  # Some high-frequency reduction
                ai_score += 4
                
            if high_freq_energy < 5:  # Very smooth frequency response
                ai_score += 8
            elif high_freq_energy < 8:  # Smooth frequency response
                ai_score += 4
        total_checks += 20
        
        # Compression Analysis (15 points) - JPEG forensics
        if compression_analysis:
            comp_uniformity = compression_analysis.get('compression_uniformity', 0)
            # AI images may have unusual compression patterns
            if comp_uniformity < 80:  # Extremely uniform compression
                ai_score += 10
            elif comp_uniformity < 120:  # Very uniform compression
                ai_score += 5
        total_checks += 15
        
        # Statistical Analysis (15 points) - Color channel analysis
        if statistical_analysis:
            channel_correlation = statistical_analysis.get('channel_correlation', 0)
            brightness_dist = statistical_analysis.get('brightness_distribution', 0)
            
            # Perfect correlations are unnatural
            if channel_correlation > 0.95:  # Near-perfect correlation
                ai_score += 10
            elif channel_correlation > 0.9:  # Very high correlation
                ai_score += 5
                
            # Check for unnatural brightness distribution
            if brightness_dist < 0.2 or brightness_dist > 0.95:
                ai_score += 5
        total_checks += 15
        
        # Noise Analysis (15 points) - Enhanced sensor noise evaluation  
        if noise_analysis:
            noise_variance = noise_analysis.get('noise_variance', 0)
            noise_pattern = noise_analysis.get('noise_pattern_score', 0)
            
            # AI images typically lack natural sensor noise
            if noise_variance < 8:  # Low noise indicates AI processing
                ai_score += 10
            elif noise_variance < 15:  # Reduced noise suggests processing
                ai_score += 6
            elif noise_variance < 25:  # Some noise reduction
                ai_score += 3
                
            if noise_pattern > 0.8:  # Regular patterns indicate AI
                ai_score += 5
            elif noise_pattern > 0.6:  # Some regularity
                ai_score += 2
        total_checks += 15
        
        # Error Level Analysis (10 points) - Professional manipulation detection
        if ela_analysis:
            ela_variance = ela_analysis.get('ela_variance', 0)
            suspicious_ratio = ela_analysis.get('suspicious_ratio', 0)
            
            # AI images typically show consistent compression patterns
            if ela_variance < 50:  # Very uniform compression response
                ai_score += 6
            elif ela_variance < 100:  # Low compression variance
                ai_score += 3
                
            if suspicious_ratio > 0.15:  # Many suspicious regions
                ai_score += 4
        total_checks += 10
        
        # Normalize score to percentage
        return min(100, max(0, ai_score))
    
    def _generate_indicators(self, exif_data, texture_analysis, frequency_analysis,
                           compression_analysis, statistical_analysis, noise_analysis):
        """Generate human-readable indicators based on improved detection"""
        indicators = []
        
        # EXIF Analysis indicators - more balanced messaging
        if not exif_data or len(exif_data) < 2:
            indicators.append('No camera metadata detected')
        elif len(exif_data) < 5:
            indicators.append('Limited image metadata found')
        elif not ('Make' in exif_data and 'Model' in exif_data):
            indicators.append('Camera details not available')
        else:
            indicators.append('Camera metadata present (authentic photo marker)')
        
        # Texture Analysis indicators
        if texture_analysis:
            texture_uniformity = texture_analysis.get('texture_uniformity', 0)
            glcm_homogeneity = texture_analysis.get('glcm_homogeneity', 0)
            edge_density = texture_analysis.get('edge_density', 0)
            
            if texture_uniformity > 0.9:
                indicators.append('Unusually uniform textures detected')
            elif texture_uniformity > 0.85:
                indicators.append('High texture uniformity observed')
                
            if glcm_homogeneity > 0.85:
                indicators.append('Very smooth texture patterns')
            elif glcm_homogeneity > 0.8:
                indicators.append('Smooth texture characteristics')
                
            if edge_density < 0.05:
                indicators.append('Unnaturally smooth edges')
            elif edge_density > 0.25:
                indicators.append('Artificial edge enhancement')
        
        # Frequency Domain indicators
        if frequency_analysis:
            freq_ratio = frequency_analysis.get('frequency_ratio', 0)
            high_freq_energy = frequency_analysis.get('high_freq_energy', 0)
            
            if freq_ratio < 0.1:
                indicators.append('Limited high-frequency content')
            elif freq_ratio < 0.2:
                indicators.append('Reduced detail frequency range')
                
            if high_freq_energy < 3:
                indicators.append('Very smooth frequency response')
            elif high_freq_energy < 6:
                indicators.append('Lower frequency complexity')
        
        # Statistical Analysis indicators
        if statistical_analysis:
            channel_correlation = statistical_analysis.get('channel_correlation', 0)
            brightness_dist = statistical_analysis.get('brightness_distribution', 0)
            
            if channel_correlation > 0.98:
                indicators.append('Perfect color correlation (AI generation)')
            elif channel_correlation > 0.92:
                indicators.append('Unnaturally high color correlation')
                
            if brightness_dist < 0.3:
                indicators.append('Artificial brightness distribution')
            elif brightness_dist > 0.9:
                indicators.append('Unnatural luminance patterns')
        
        # Noise Analysis indicators
        if noise_analysis:
            noise_variance = noise_analysis.get('noise_variance', 0)
            noise_pattern = noise_analysis.get('noise_pattern_score', 0)
            
            if noise_variance < 5:
                indicators.append('Missing camera sensor noise (AI)')
            elif noise_variance < 15:
                indicators.append('Unusually low image noise')
                
            if noise_pattern < 0.1:
                indicators.append('Overly regular noise patterns')
        
        # Compression indicators
        if compression_analysis:
            comp_uniformity = compression_analysis.get('compression_uniformity', 0)
            if comp_uniformity < 50:
                indicators.append('Artificial compression uniformity')
            elif comp_uniformity < 150:
                indicators.append('Suspicious compression patterns')
        
        # Ensure we always return some indicators
        return indicators[:8] if indicators else ['Analysis completed - no strong indicators found']


def main():
    """Main function for command-line usage"""
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Usage: python simple_ai_detector.py <image_path>'}))
        sys.exit(1)
    
    try:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(json.dumps({'error': 'Image file not found'}))
            sys.exit(1)
        
        # Read image file
        with open(image_path, 'rb') as f:
            image_buffer = f.read()
        
        # Initialize detector and analyze
        detector = SimpleAIImageDetector()
        result = detector.detect_ai_image(image_buffer)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': f'Analysis failed: {str(e)}'}))
        sys.exit(1)

if __name__ == '__main__':
    main()