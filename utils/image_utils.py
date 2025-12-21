"""
Image Utility Functions
=======================

Helper functions for image processing operations used in HDR pipeline.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path


def simulate_multi_exposure_capture(base_image: np.ndarray,
                                   exposure_ratios: List[float] = None) -> Tuple[List[np.ndarray], List[float]]:
    """
    Simulate multi-frame exposure capture by applying different exposure adjustments.
    
    In real HDR photography, you would capture multiple images at different
    exposure settings. This function simulates that process by scaling pixel
    values to represent different exposure times.
    
    The relationship between pixel value and exposure time:
    - Higher exposure time → brighter image (more light captured)
    - Lower exposure time → darker image (less light captured)
    
    We simulate this by:
    - Overexposed (ratio > 1.0): multiply pixel values by ratio
    - Underexposed (ratio < 1.0): divide pixel values by (1/ratio)
    
    Args:
        base_image: Input image (assumed to be normally exposed, uint8)
        exposure_ratios: List of exposure time ratios relative to base exposure
                        Default: [0.25, 0.5, 1.0, 2.0, 4.0] (±2 stops)
                        
    Returns:
        Tuple of (exposure_images, exposure_times)
        - exposure_images: List of simulated exposure images (uint8)
        - exposure_times: List of exposure time ratios
    """
    if exposure_ratios is None:
        exposure_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    print("📸 Simulating multi-exposure capture...")
    
    exposure_images = []
    exposure_times = []
    
    for ratio in exposure_ratios:
        # Convert to float for calculations
        img_float = base_image.astype(np.float32)
        
        # Apply exposure adjustment
        if ratio > 1.0:
            # Overexposed: multiply by ratio (brighter)
            adjusted = img_float * ratio
        else:
            # Underexposed: divide by (1/ratio) (darker)
            adjusted = img_float / (1.0 / ratio)
        
        # Clip to valid range [0, 255] and convert back to uint8
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        exposure_images.append(adjusted)
        exposure_times.append(ratio)
    
    print(f"   Generated {len(exposure_images)} exposure images with ratios: {exposure_ratios}")
    return exposure_images, exposure_times


def save_hdr_image(hdr_image: np.ndarray, file_path: Path, prefer_exr: bool = True) -> bool:
    """
    Save HDR image to file with multiple format fallback options.
    
    Tries to save in the following order:
    1. EXR format (if OpenEXR is available in OpenCV)
    2. NumPy binary format (.npy) - always works, preserves full precision
    3. PNG format with normalized float32 values
    
    Args:
        hdr_image: HDR image to save (float32, 0-1 range)
        file_path: Path object where to save the file
        prefer_exr: If True, try EXR first (default: True)
        
    Returns:
        True if file was saved successfully, False otherwise
    """
    # Ensure file_path is a Path object
    file_path = Path(file_path)
    
    # Try EXR format first (if preferred and file extension is .exr)
    if prefer_exr and file_path.suffix.lower() == '.exr':
        try:
            # Try saving as EXR
            success = cv2.imwrite(str(file_path), hdr_image)
            if success:
                print(f"💾 HDR image saved to: {file_path} (EXR format)")
                return True
        except Exception as e:
            # EXR not available, fall through to alternatives
            pass
    
    # Fallback 1: Save as NumPy binary format (.npy)
    # This preserves full precision and is always available
    npy_path = file_path.with_suffix('.npy')
    try:
        np.save(str(npy_path), hdr_image)
        print(f"💾 HDR image saved to: {npy_path} (NumPy format)")
        print(f"   Note: EXR format not available. Use: hdr_image = np.load('{npy_path}') to load")
        
        # Also try to save a normalized PNG version for preview
        png_path = file_path.with_suffix('.png')
        try:
            # Normalize to 0-65535 range for 16-bit PNG
            hdr_normalized = (hdr_image * 65535.0).astype(np.uint16)
            cv2.imwrite(str(png_path), hdr_normalized)
            print(f"💾 HDR preview saved to: {png_path} (16-bit PNG, normalized)")
        except Exception:
            pass
        
        return True
    except Exception as e:
        print(f"⚠️  Could not save HDR image: {e}")
        return False
