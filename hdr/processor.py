"""
HDR Processor - Main Pipeline Orchestrator
==========================================

Implements the complete HDR processing pipeline based on Chapter 13 of
"Mobile Computational Photography".

The pipeline consists of 5 main stages:
1. Multi-frame exposure capture (simulation)
2. Image alignment (feature-based registration)
3. Weight computation (triangle weighting function)
4. HDR fusion (weighted averaging)
5. Tone mapping (HDR to LDR conversion)

This class orchestrates all the components to process HDR images from
multiple exposures.
"""

from typing import List, Dict, Optional
import numpy as np

from hdr.alignment import ImageAligner
from hdr.fusion import HDRFusion
from hdr.tone_mapping import ToneMapper
from utils.image_utils import simulate_multi_exposure_capture


class HDRProcessor:
    """
    Main HDR processing pipeline orchestrator.
    
    This class coordinates all stages of the HDR pipeline:
    - Multi-exposure simulation
    - Image alignment
    - HDR fusion
    - Tone mapping
    
    Usage:
        processor = HDRProcessor()
        results = processor.process_hdr_pipeline(image)
    """
    
    def __init__(self, n_features: int = 1000, match_ratio: float = 0.7):
        """
        Initialize the HDR processor.
        
        Args:
            n_features: Number of features to detect for alignment (default: 1000)
            match_ratio: Feature matching ratio threshold (default: 0.7)
        """
        # Initialize component modules
        self.aligner = ImageAligner(n_features=n_features, match_ratio=match_ratio)
        self.fusion = HDRFusion()
        self.tone_mapper = ToneMapper()
        
        # Internal state (can be accessed after processing)
        self.exposure_images: Optional[List[np.ndarray]] = None
        self.exposure_times: Optional[List[float]] = None
        self.aligned_images: Optional[List[np.ndarray]] = None
        self.weights: Optional[List[np.ndarray]] = None
        self.hdr_image: Optional[np.ndarray] = None
        self.tone_mapped: Optional[np.ndarray] = None
    
    def process_hdr_pipeline(self,
                           base_image: np.ndarray,
                           exposure_ratios: Optional[List[float]] = None,
                           tone_mapping_method: str = 'reinhard',
                           reference_idx: int = 2,
                           **tone_mapping_kwargs) -> Dict:
        """
        Execute the complete HDR processing pipeline.
        
        Pipeline stages:
        1. Simulate multi-exposure capture
        2. Align all exposures to a reference image
        3. Compute fusion weights for each exposure
        4. Fuse exposures into HDR image
        5. Apply tone mapping to convert HDR to displayable format
        
        Args:
            base_image: Input image (uint8, BGR format from OpenCV)
            exposure_ratios: List of exposure time ratios
                           Default: [0.25, 0.5, 1.0, 2.0, 4.0] (±2 stops)
            tone_mapping_method: 'reinhard' or 'drago' (default: 'reinhard')
            reference_idx: Index of reference image for alignment (default: 2)
            **tone_mapping_kwargs: Additional parameters for tone mapping
                                  (e.g., key_value for Reinhard, bias for Drago)
        
        Returns:
            Dictionary containing all processing results:
            {
                'exposure_images': List of exposure images,
                'exposure_times': List of exposure time ratios,
                'aligned_images': List of aligned images,
                'weights': List of weight maps,
                'hdr_image': HDR image (float32, 0-1 range),
                'tone_mapped': Tone-mapped LDR image (uint8, 0-255 range)
            }
        """
        if exposure_ratios is None:
            exposure_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
        
        print("🚀 Starting HDR Processing Pipeline")
        print("=" * 60)
        print(f"   Input image shape: {base_image.shape}")
        print(f"   Exposure ratios: {exposure_ratios}")
        print(f"   Tone mapping: {tone_mapping_method}")
        print("=" * 60)
        
        # Stage 1: Simulate multi-exposure capture
        # In real photography, this would be actual camera captures
        self.exposure_images, self.exposure_times = simulate_multi_exposure_capture(
            base_image, exposure_ratios
        )
        
        # Stage 2: Align images to compensate for camera shake/motion
        # This is critical to avoid ghosting artifacts in the final HDR
        self.aligned_images = self.aligner.align_images(
            self.exposure_images, reference_idx=reference_idx
        )
        
        # Stage 3: Compute fusion weights
        # Weight function determines how much to trust each pixel in each exposure
        self.weights = self.fusion.compute_weights(self.aligned_images)
        
        # Stage 4: Fuse exposures into HDR image
        # Combines all exposures using weighted averaging
        self.hdr_image = self.fusion.fuse_hdr(
            self.aligned_images,
            self.exposure_times,
            self.weights
        )
        
        # Stage 5: Apply tone mapping
        # Converts HDR (high dynamic range) to LDR (low dynamic range) for display
        self.tone_mapped = self.tone_mapper.apply_tone_mapping(
            self.hdr_image,
            method=tone_mapping_method,
            **tone_mapping_kwargs
        )
        
        print("=" * 60)
        print("✅ HDR Processing Pipeline Completed")
        print(f"   Final image shape: {self.tone_mapped.shape}")
        print("=" * 60)
        
        # Return all results for further processing or visualization
        return {
            'exposure_images': self.exposure_images,
            'exposure_times': self.exposure_times,
            'aligned_images': self.aligned_images,
            'weights': self.weights,
            'hdr_image': self.hdr_image,
            'tone_mapped': self.tone_mapped
        }
