"""
HDR Fusion Module
=================

Implements HDR image fusion from multiple aligned exposures.
Based on Chapter 13 of "Mobile Computational Photography".

The fusion process combines information from multiple exposures:
1. Compute weights for each pixel in each exposure (triangle weighting)
2. Apply exposure compensation to normalize pixel values
3. Weighted average fusion to combine the best parts of each exposure

Key concept: Well-exposed pixels (not too dark, not too bright) get higher weights.
"""

import numpy as np
from typing import List


class HDRFusion:
    """
    HDR image fusion from multiple aligned exposures.
    
    Uses weighted averaging with triangle weighting function to combine
    multiple exposure images into a single HDR image.
    """
    
    @staticmethod
    def compute_weights(images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute weighting function for HDR fusion.
        
        Uses triangle weighting function that gives higher weight to pixels
        that are well-exposed (around 50% gray value). This ensures we trust
        pixels that are not clipped (overexposed) or too noisy (underexposed).
        
        Triangle weighting function:
        - w(I) = 2*I for I <= 0.5 (increasing weight from 0 to 1)
        - w(I) = 2*(1-I) for I > 0.5 (decreasing weight from 1 to 0)
        
        Peak weight is at I = 0.5 (middle gray).
        
        Args:
            images: List of aligned exposure images (uint8, 0-255)
            
        Returns:
            List of weight maps (float32, 0-1 range), one per input image
        """
        print("⚖️  Computing fusion weights...")
        
        weights = []
        for img in images:
            # Convert to float and normalize to [0, 1] range
            img_float = img.astype(np.float32) / 255.0
            
            # Triangle weighting: higher weight for mid-tone values
            # For values <= 0.5: weight increases linearly from 0 to 1
            # For values > 0.5: weight decreases linearly from 1 to 0
            weight = np.where(
                img_float <= 0.5,
                2 * img_float,              # Linear increase: 0 → 1
                2 * (1 - img_float)         # Linear decrease: 1 → 0
            )
            
            # For color images, compute single weight map from all channels
            if len(img.shape) == 3:
                # Average across color channels for consistent weighting
                weight = np.mean(weight, axis=2, keepdims=True)
                # Replicate for all channels
                weight = np.repeat(weight, img.shape[2], axis=2)
            
            weights.append(weight)
        
        print(f"   Computed weights for {len(weights)} images")
        return weights
    
    @staticmethod
    def fuse_hdr(aligned_images: List[np.ndarray],
                exposure_times: List[float],
                weights: List[np.ndarray]) -> np.ndarray:
        """
        Fuse aligned exposure images into a single HDR image.
        
        The fusion process:
        1. Normalize each image by its exposure time (exposure compensation)
        2. Apply weights to emphasize well-exposed pixels
        3. Weighted average to combine all exposures
        
        Formula: HDR(x,y) = Σ[I_i(x,y) * w_i(x,y) / t_i] / Σ[w_i(x,y)]
        where:
        - I_i is the pixel value in image i
        - w_i is the weight for image i
        - t_i is the exposure time ratio for image i
        
        Args:
            aligned_images: List of aligned exposure images (uint8, 0-255)
            exposure_times: Corresponding exposure time ratios (relative to base)
            weights: Weight maps for each image (float32, 0-1 range)
            
        Returns:
            HDR image (float32, 0-1 range, high dynamic range)
        """
        print("🔗 Fusing images into HDR...")
        
        # Initialize accumulator for weighted sum and weight sum
        hdr = np.zeros_like(aligned_images[0], dtype=np.float32)
        weight_sum = np.zeros_like(aligned_images[0], dtype=np.float32)
        
        # Process each exposure image
        for img, exposure_time, weight in zip(aligned_images, exposure_times, weights):
            # Convert to float and normalize to [0, 1] range
            img_float = img.astype(np.float32) / 255.0
            
            # Exposure compensation: divide by exposure time to recover radiance
            # Higher exposure time → brighter image → divide by larger value
            # This normalizes all images to the same radiance scale
            compensated = img_float / exposure_time
            
            # Apply weights and accumulate
            weighted_img = compensated * weight
            hdr += weighted_img
            weight_sum += weight
        
        # Avoid division by zero (add small epsilon)
        weight_sum = np.maximum(weight_sum, 1e-8)
        
        # Normalize by total weight to get final HDR image
        hdr = hdr / weight_sum
        
        # Ensure values are in valid range [0, 1]
        hdr = np.clip(hdr, 0, 1)
        
        print("   HDR fusion completed")
        return hdr
