"""
Tone Mapping Module
===================

Implements tone mapping algorithms to convert HDR images to displayable LDR.
Based on Chapter 13 of "Mobile Computational Photography".

Tone mapping is essential because:
- HDR images have a much larger dynamic range than displays can show
- We need to compress the HDR range to fit within 0-255 for standard displays
- Must preserve visual quality and detail

Available algorithms:
- Reinhard: Physically-based, preserves local contrast
- Drago: Optimized for display devices, better color preservation
"""

import numpy as np
import cv2
from typing import Literal


class ToneMapper:
    """
    Tone mapping algorithms for HDR to LDR conversion.
    """
    
    @staticmethod
    def tone_map_reinhard(hdr_image: np.ndarray, key_value: float = 0.18) -> np.ndarray:
        """
        Apply Reinhard tone mapping operator.
        
        Reinhard tone mapping is a simple and effective global tone mapping
        algorithm that preserves local contrast while compressing the dynamic range.
        
        Formula: Ld = L * (1 + L/Lw²) / (1 + L)
        where:
        - Ld: Display luminance (output)
        - L: Scene luminance (input)
        - Lw: World adaptation luminance (key_value, typically 0.18 for middle gray)
        
        The algorithm has a smooth roll-off at both dark and bright regions,
        preventing clipping while maintaining detail.
        
        Args:
            hdr_image: HDR image (float32, 0-1 range)
            key_value: Key value for tone mapping, represents middle gray
                      (default: 0.18, standard for photographic tone mapping)
                      
        Returns:
            Tone-mapped LDR image (uint8, 0-255 range)
        """
        print("🎨 Applying Reinhard tone mapping...")
        
        # Compute luminance channel (for perceptual tone mapping)
        if len(hdr_image.shape) == 3:
            # Color image: use standard luminance weights (ITU-R BT.601)
            # Y = 0.299*R + 0.587*G + 0.114*B
            luminance = (
                0.299 * hdr_image[:, :, 0] +
                0.587 * hdr_image[:, :, 1] +
                0.114 * hdr_image[:, :, 2]
            )
        else:
            # Grayscale image: use directly
            luminance = hdr_image
        
        # Avoid division by zero
        luminance = np.maximum(luminance, 1e-8)
        
        # Apply Reinhard tone mapping formula
        # The formula compresses high values while preserving low values
        tone_mapped = hdr_image * (1 + hdr_image / (key_value**2)) / (
            1 + luminance[..., np.newaxis] if len(hdr_image.shape) == 3 else hdr_image
        )
        
        # Convert to displayable range [0, 255]
        tone_mapped = np.clip(tone_mapped * 255, 0, 255).astype(np.uint8)
        
        print("   Tone mapping completed")
        return tone_mapped
    
    @staticmethod
    def tone_map_drago(hdr_image: np.ndarray, 
                      bias: float = 0.85,
                      gamma: float = 2.2,
                      saturation: float = 1.0) -> np.ndarray:
        """
        Apply Drago tone mapping using OpenCV implementation.
        
        Drago tone mapping is optimized for display devices and provides
        good color preservation. It uses adaptive logarithmic mapping that
        adjusts based on the content of the image.
        
        Args:
            hdr_image: HDR image (float32, 0-1 range, or higher)
            bias: Bias parameter controlling the compression curve
                  (0-1, default: 0.85, higher = brighter)
            gamma: Gamma correction value (default: 2.2, standard display gamma)
            saturation: Color saturation factor (default: 1.0, no change)
                        >1.0 increases saturation, <1.0 decreases
                       
        Returns:
            Tone-mapped LDR image (uint8, 0-255 range)
        """
        print("🎨 Applying Drago tone mapping...")
        
        # Normalize to [0, 1] range if needed
        if hdr_image.max() > 1.0:
            hdr_normalized = hdr_image / hdr_image.max()
        else:
            hdr_normalized = hdr_image.copy()
        
        # Create Drago tone mapper using OpenCV
        tonemap = cv2.createTonemapDrago(
            gamma=gamma,
            saturation=saturation,
            bias=bias
        )
        
        # Apply tone mapping
        tone_mapped = tonemap.process(hdr_normalized)
        
        # Convert to displayable range [0, 255]
        tone_mapped = np.clip(tone_mapped * 255, 0, 255).astype(np.uint8)
        
        print("   Drago tone mapping completed")
        return tone_mapped
    
    @staticmethod
    def apply_tone_mapping(hdr_image: np.ndarray,
                          method: Literal['reinhard', 'drago'] = 'reinhard',
                          **kwargs) -> np.ndarray:
        """
        Apply tone mapping using specified method.
        
        Args:
            hdr_image: HDR image to tone map (float32, 0-1 range)
            method: Tone mapping method ('reinhard' or 'drago')
            **kwargs: Additional parameters for tone mapping algorithms
            
        Returns:
            Tone-mapped LDR image (uint8, 0-255 range)
        """
        if method == 'drago':
            return ToneMapper.tone_map_drago(hdr_image, **kwargs)
        else:
            return ToneMapper.tone_map_reinhard(
                hdr_image, 
                key_value=kwargs.get('key_value', 0.18)
            )

