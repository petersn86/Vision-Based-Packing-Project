##############################################
# @Author: Peter Nolan
# @Document: 'image_preprocessor.py'
#
# Description:
# Image preprocessing pipeline to improve YOLO detection quality.
# Applies sharpening, contrast enhancement, denoising, and other
# techniques to make objects more detectable.
#
##############################################

import cv2
import numpy as np
from typing import Optional, Tuple


class ImagePreprocessor:
    """
    Preprocesses images to improve YOLO detection quality.
    
    Techniques:
    - Sharpening (enhances edges)
    - Contrast enhancement (CLAHE)
    - Denoising (reduces noise)
    - Brightness/gamma correction
    - Upscaling (for small objects)
    """
    
    def __init__(self,
                 sharpen: bool = True,
                 enhance_contrast: bool = True,
                 denoise: bool = True,
                 auto_brightness: bool = True,
                 upscale_factor: float = 1.0):
        """
        Initialize preprocessor
        
        Args:
            sharpen: Apply unsharp masking for edge enhancement
            enhance_contrast: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            denoise: Apply denoising filter
            auto_brightness: Auto-adjust brightness/gamma
            upscale_factor: Upscale image (e.g., 1.5 = 150% size) - helps with small objects
        """
        self.sharpen = sharpen
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        self.auto_brightness = auto_brightness
        self.upscale_factor = upscale_factor
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline
        
        Args:
            frame: Input BGR image
            
        Returns:
            Preprocessed BGR image
        """
        processed = frame.copy()
        
        # 1. Upscale if needed (do first to preserve detail)
        if self.upscale_factor > 1.0:
            processed = self._upscale(processed)
        
        # 2. Denoise (reduces noise before other operations)
        if self.denoise:
            processed = self._denoise(processed)
        
        # 3. Auto brightness/gamma correction
        if self.auto_brightness:
            processed = self._auto_brightness(processed)
        
        # 4. Contrast enhancement (CLAHE)
        if self.enhance_contrast:
            processed = self._enhance_contrast(processed)
        
        # 5. Sharpening (do last to enhance final edges)
        if self.sharpen:
            processed = self._sharpen(processed)
        
        return processed
    
    # ========================================
    # Individual preprocessing techniques
    # ========================================
    
    def _sharpen(self, frame: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Unsharp masking - makes edges crisp and objects more distinct
        
        Args:
            frame: Input image
            strength: Sharpening strength (1.0-3.0, higher = sharper)
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(frame, (0, 0), 3.0)
        
        # Subtract blur from original and add back
        sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
        
        return sharpened
    
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        CLAHE - Adaptive histogram equalization
        Makes objects stand out from background
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Non-local means denoising
        Reduces camera noise while preserving edges
        """
        denoised = cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=10,           # Filter strength (10 is good for typical noise)
            hColor=10,      # Color component filter strength
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised
    
    def _auto_brightness(self, frame: np.ndarray) -> np.ndarray:
        """
        Auto-adjust brightness using gamma correction
        Helps with underexposed or overexposed images
        """
        # Convert to grayscale to measure brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Target brightness (128 = middle gray)
        target = 128
        
        if abs(mean_brightness - target) < 20:
            # Already well-exposed
            return frame
        
        # Calculate gamma correction factor
        # Gamma < 1 brightens, Gamma > 1 darkens
        gamma = np.log(target / 255) / np.log(mean_brightness / 255)
        
        # Clamp gamma to reasonable range
        gamma = np.clip(gamma, 0.5, 2.0)
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        
        # Apply gamma correction
        corrected = cv2.LUT(frame, table)
        
        return corrected
    
    def _upscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale image using high-quality interpolation
        Helps YOLO detect small objects better
        """
        h, w = frame.shape[:2]
        new_h = int(h * self.upscale_factor)
        new_w = int(w * self.upscale_factor)
        
        # Use INTER_CUBIC for upscaling (high quality)
        upscaled = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )
        
        return upscaled
    
    # ========================================
    # Additional specialized techniques
    # ========================================
    
    def apply_edge_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """
        Strong edge enhancement using Laplacian
        Use when objects blend into background
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Add edges back to original
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(frame, 1.0, 
                                   cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR), 
                                   0.3, 0)
        
        return enhanced
    
    def apply_color_pop(self, frame: np.ndarray, saturation_boost: float = 1.3) -> np.ndarray:
        """
        Boost color saturation to make colored objects more distinct
        Helps with colorful items (screwdrivers, bottles, etc.)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Boost saturation
        s = s * saturation_boost
        s = np.clip(s, 0, 255)
        
        # Merge and convert back
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        boosted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return boosted


# ========================================
# Preset configurations
# ========================================

def get_preset_config(preset: str) -> dict:
    """
    Get preprocessor configuration presets
    
    Presets:
        'light' - Minimal processing, fast
        'standard' - Balanced quality/speed (RECOMMENDED)
        'aggressive' - Maximum enhancement, slower
        'small_objects' - Optimized for detecting small/distant items
        'low_light' - Optimized for dark/underexposed images
    """
    presets = {
        'light': {
            'sharpen': True,
            'enhance_contrast': False,
            'denoise': False,
            'auto_brightness': False,
            'upscale_factor': 1.0
        },
        'standard': {
            'sharpen': True,
            'enhance_contrast': True,
            'denoise': True,
            'auto_brightness': True,
            'upscale_factor': 1.0
        },
        'aggressive': {
            'sharpen': True,
            'enhance_contrast': True,
            'denoise': True,
            'auto_brightness': True,
            'upscale_factor': 1.2
        },
        'small_objects': {
            'sharpen': True,
            'enhance_contrast': True,
            'denoise': True,
            'auto_brightness': True,
            'upscale_factor': 1.5  # Bigger images help detect small objects
        },
        'low_light': {
            'sharpen': False,  # Don't sharpen noisy images
            'enhance_contrast': True,
            'denoise': True,   # Critical for low light
            'auto_brightness': True,
            'upscale_factor': 1.0
        }
    }
    
    return presets.get(preset, presets['standard'])


# ========================================
# Convenience function
# ========================================

def enhance_for_detection(frame: np.ndarray, 
                          preset: str = 'standard',
                          custom_config: dict = None) -> np.ndarray:
    """
    One-line function to enhance image for YOLO detection
    
    Args:
        frame: Input BGR image
        preset: Preset name ('light', 'standard', 'aggressive', 'small_objects', 'low_light')
        custom_config: Custom config dict (overrides preset)
        
    Returns:
        Enhanced image ready for YOLO
        
    Example:
        enhanced = enhance_for_detection(frame, preset='aggressive')
        detections = yolo.detectObjects(enhanced, ...)
    """
    config = get_preset_config(preset)
    
    if custom_config:
        config.update(custom_config)
    
    preprocessor = ImagePreprocessor(**config)
    return preprocessor.preprocess(frame)