#!/usr/bin/env python3
"""
HDR (High Dynamic Range) Image Processing Example
=================================================

This example demonstrates the complete HDR pipeline including:
1. Multi-frame exposure capture simulation
2. Image alignment using feature matching
3. HDR fusion using weighted averaging
4. Tone mapping for display

Based on the mobile computational photography principles outlined in design.txt
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import argparse
import os
from pathlib import Path


class HDRProcessor:
    """
    HDR Image Processing Pipeline
    
    Implements the core HDR workflow as described in design.txt:
    - Multi-frame compositing
    - Image alignment
    - Dynamic range fusion
    - Post-processing optimization
    """
    
    def __init__(self):
        self.aligned_images: List[np.ndarray] = []
        self.exposure_times: List[float] = []
        self.hdr_image: Optional[np.ndarray] = None
        
    def simulate_multi_exposure_capture(self, base_image: np.ndarray, 
                                      exposure_ratios: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Simulate multi-frame exposure capture by applying different exposure adjustments
        
        Args:
            base_image: Input image (assumed to be normally exposed)
            exposure_ratios: List of exposure time ratios relative to base exposure
            
        Returns:
            Tuple of (exposure_images, exposure_times)
        """
        print("📸 Simulating multi-exposure capture...")
        
        exposure_images = []
        exposure_times = []
        
        for ratio in exposure_ratios:
            # Simulate different exposure times by scaling pixel values
            # Higher exposure = brighter image (multiply by ratio)
            # Lower exposure = darker image (divide by ratio)
            
            if ratio > 1.0:
                # Overexposed simulation
                adjusted = np.clip(base_image.astype(np.float32) * ratio, 0, 255)
            else:
                # Underexposed simulation
                adjusted = np.clip(base_image.astype(np.float32) / (1.0 / ratio), 0, 255)
            
            exposure_images.append(adjusted.astype(np.uint8))
            exposure_times.append(ratio)
            
        print(f"   Generated {len(exposure_images)} exposure images with ratios: {exposure_ratios}")
        return exposure_images, exposure_times
    
    def align_images(self, images: List[np.ndarray], reference_idx: int = 2) -> List[np.ndarray]:
        """
        Align multiple exposure images using feature-based registration
        
        Args:
            images: List of exposure images to align
            reference_idx: Index of reference image for alignment
            
        Returns:
            List of aligned images
        """
        print("🎯 Performing image alignment...")
        
        if len(images) < 2:
            return images
            
        reference = images[reference_idx]
        aligned_images = [reference.copy()]
        
        # Use ORB detector for feature matching
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find reference keypoints and descriptors
        kp_ref, des_ref = orb.detectAndCompute(reference, None)
        
        if des_ref is None:
            print("   Warning: No features detected in reference image")
            return images
        
        for i, img in enumerate(images):
            if i == reference_idx:
                continue
                
            # Find keypoints and descriptors for current image
            kp_curr, des_curr = orb.detectAndCompute(img, None)
            
            if des_curr is None:
                print(f"   Warning: No features detected in image {i}")
                aligned_images.append(img)
                continue
            
            # Match features using FLANN matcher
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            try:
                matches = flann.knnMatch(des_ref, des_curr, k=2)
            except:
                print(f"   Warning: Feature matching failed for image {i}")
                aligned_images.append(img)
                continue
            
            # Apply ratio test to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                print(f"   Warning: Insufficient good matches ({len(good_matches)}) for image {i}")
                aligned_images.append(img)
                continue
            
            # Extract matched keypoints
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Estimate homography matrix
            try:
                homography, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                if homography is not None:
                    # Apply homography transformation
                    h, w = reference.shape[:2]
                    aligned = cv2.warpPerspective(img, homography, (w, h))
                    aligned_images.append(aligned)
                    print(f"   Successfully aligned image {i} with {len(good_matches)} matches")
                else:
                    print(f"   Warning: Homography estimation failed for image {i}")
                    aligned_images.append(img)
                    
            except Exception as e:
                print(f"   Warning: Alignment failed for image {i}: {e}")
                aligned_images.append(img)
        
        print(f"   Completed alignment for {len(aligned_images)} images")
        return aligned_images
    
    def compute_weights(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute weighting function for HDR fusion
        
        Uses triangle weighting function that gives higher weight to 
        pixels that are not too dark or too bright
        
        Args:
            images: List of aligned exposure images
            
        Returns:
            List of weight maps for each image
        """
        print("⚖️  Computing fusion weights...")
        
        weights = []
        for img in images:
            # Convert to float and normalize to [0, 1]
            img_float = img.astype(np.float32) / 255.0
            
            # Triangle weighting function
            # Higher weight for mid-tone values (around 0.5)
            weight = np.where(img_float <= 0.5, 
                            2 * img_float,  # Linear increase from 0 to 1
                            2 * (1 - img_float))  # Linear decrease from 1 to 0
            
            # Apply to all channels if color image
            if len(img.shape) == 3:
                weight = np.mean(weight, axis=2, keepdims=True)
                weight = np.repeat(weight, img.shape[2], axis=2)
            
            weights.append(weight)
        
        print(f"   Computed weights for {len(weights)} images")
        return weights
    
    def fuse_hdr(self, aligned_images: List[np.ndarray], 
                exposure_times: List[float], weights: List[np.ndarray]) -> np.ndarray:
        """
        Fuse aligned exposure images into HDR image
        
        Args:
            aligned_images: List of aligned exposure images
            exposure_times: Corresponding exposure time ratios
            weights: Weight maps for each image
            
        Returns:
            HDR image (high dynamic range, typically in float32)
        """
        print("🔗 Fusing images into HDR...")
        
        # Initialize HDR image
        hdr = np.zeros_like(aligned_images[0], dtype=np.float32)
        weight_sum = np.zeros_like(aligned_images[0], dtype=np.float32)
        
        for img, exposure_time, weight in zip(aligned_images, exposure_times, weights):
            # Convert to float and normalize
            img_float = img.astype(np.float32) / 255.0
            
            # Apply exposure compensation
            # Higher exposure time means brighter image, so we divide by it
            compensated = img_float / exposure_time
            
            # Apply weights and accumulate
            weighted_img = compensated * weight
            hdr += weighted_img
            weight_sum += weight
        
        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-8)
        hdr = hdr / weight_sum
        
        # Convert to proper HDR range (typically 0-1 or higher)
        hdr = np.clip(hdr, 0, 1)
        
        print("   HDR fusion completed")
        return hdr
    
    def tone_map_reinhard(self, hdr_image: np.ndarray, key_value: float = 0.18) -> np.ndarray:
        """
        Apply Reinhard tone mapping to convert HDR to LDR
        
        Args:
            hdr_image: HDR image (float, 0-1 range)
            key_value: Key value for tone mapping (typical: 0.18)
            
        Returns:
            Tone-mapped LDR image (uint8, 0-255)
        """
        print("🎨 Applying Reinhard tone mapping...")
        
        # Reinhard tone mapping formula
        # Ld = L * (1 + L/Lw^2) / (1 + L)
        # where Lw is the world luminance (key value)
        
        # Calculate luminance
        if len(hdr_image.shape) == 3:
            # Color image - use luminance channel
            luminance = 0.299 * hdr_image[:, :, 0] + 0.587 * hdr_image[:, :, 1] + 0.114 * hdr_image[:, :, 2]
        else:
            # Grayscale image
            luminance = hdr_image
        
        # Avoid division by zero
        luminance = np.maximum(luminance, 1e-8)
        
        # Apply Reinhard tone mapping
        tone_mapped = hdr_image * (1 + hdr_image / (key_value**2)) / (1 + luminance[..., np.newaxis] if len(hdr_image.shape) == 3 else hdr_image)
        
        # Convert to 0-255 range
        tone_mapped = np.clip(tone_mapped * 255, 0, 255).astype(np.uint8)
        
        print("   Tone mapping completed")
        return tone_mapped
    
    def tone_map_drago(self, hdr_image: np.ndarray, bias: float = 0.85) -> np.ndarray:
        """
        Apply Drago tone mapping using OpenCV
        
        Args:
            hdr_image: HDR image (float, 0-1 range)
            bias: Bias parameter (0-1, typical: 0.85)
            
        Returns:
            Tone-mapped LDR image (uint8, 0-255)
        """
        print("🎨 Applying Drago tone mapping...")
        
        # Convert to 0-1 range if needed
        if hdr_image.max() > 1.0:
            hdr_normalized = hdr_image / hdr_image.max()
        else:
            hdr_normalized = hdr_image.copy()
        
        # Apply Drago tone mapping
        tonemap = cv2.createTonemapDrago(gamma=2.2, saturation=1.0, bias=bias)
        tone_mapped = tonemap.process(hdr_normalized)
        
        # Convert to 0-255 range
        tone_mapped = np.clip(tone_mapped * 255, 0, 255).astype(np.uint8)
        
        print("   Drago tone mapping completed")
        return tone_mapped
    
    def process_hdr_pipeline(self, base_image: np.ndarray, 
                           exposure_ratios: List[float] = None,
                           tone_mapping_method: str = 'reinhard') -> dict:
        """
        Complete HDR processing pipeline
        
        Args:
            base_image: Input image
            exposure_ratios: List of exposure ratios (default: [0.25, 0.5, 1.0, 2.0, 4.0])
            tone_mapping_method: 'reinhard' or 'drago'
            
        Returns:
            Dictionary containing all processing results
        """
        if exposure_ratios is None:
            exposure_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
        
        print("🚀 Starting HDR Processing Pipeline")
        print("=" * 50)
        
        # Step 1: Simulate multi-exposure capture
        exposure_images, exposure_times = self.simulate_multi_exposure_capture(base_image, exposure_ratios)
        
        # Step 2: Align images
        aligned_images = self.align_images(exposure_images)
        
        # Step 3: Compute fusion weights
        weights = self.compute_weights(aligned_images)
        
        # Step 4: Fuse into HDR
        hdr_image = self.fuse_hdr(aligned_images, exposure_times, weights)
        
        # Step 5: Tone mapping
        if tone_mapping_method == 'drago':
            tone_mapped = self.tone_map_drago(hdr_image)
        else:
            tone_mapped = self.tone_map_reinhard(hdr_image)
        
        print("=" * 50)
        print("✅ HDR Processing Pipeline Completed")
        
        return {
            'exposure_images': exposure_images,
            'exposure_times': exposure_times,
            'aligned_images': aligned_images,
            'weights': weights,
            'hdr_image': hdr_image,
            'tone_mapped': tone_mapped
        }


def visualize_results(results: dict, save_path: str = None):
    """
    Visualize HDR processing results
    
    Args:
        results: Dictionary from process_hdr_pipeline
        save_path: Optional path to save visualization
    """
    print("📊 Creating visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('HDR Processing Pipeline Results', fontsize=16)
    
    # Show exposure images (first 3)
    for i in range(min(3, len(results['exposure_images']))):
        img = results['exposure_images'][i]
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        axes[0, i].imshow(img_rgb, cmap='gray' if len(img.shape) == 2 else None)
        axes[0, i].set_title(f'Exposure {results["exposure_times"][i]:.2f}x')
        axes[0, i].axis('off')
    
    # Show weight maps (first 3)
    for i in range(min(3, len(results['weights']))):
        weight = results['weights'][i]
        if len(weight.shape) == 3:
            weight = weight[:, :, 0]  # Use first channel for grayscale display
        axes[1, i].imshow(weight, cmap='hot')
        axes[1, i].set_title(f'Weight Map {i+1}')
        axes[1, i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved to: {save_path}")
    
    plt.show()
    
    # Show HDR and tone-mapped results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # HDR image (log scale for better visualization)
    hdr_log = np.log(results['hdr_image'] + 1e-8)
    hdr_log = (hdr_log - hdr_log.min()) / (hdr_log.max() - hdr_log.min())
    
    if len(results['hdr_image'].shape) == 3:
        hdr_rgb = cv2.cvtColor((hdr_log * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        axes[0].imshow(hdr_rgb)
    else:
        axes[0].imshow(hdr_log, cmap='gray')
    axes[0].set_title('HDR Image (Log Scale)')
    axes[0].axis('off')
    
    # Tone-mapped result
    tone_mapped = results['tone_mapped']
    if len(tone_mapped.shape) == 3:
        tone_mapped_rgb = cv2.cvtColor(tone_mapped, cv2.COLOR_BGR2RGB)
        axes[1].imshow(tone_mapped_rgb)
    else:
        axes[1].imshow(tone_mapped, cmap='gray')
    axes[1].set_title('Tone-Mapped Result')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        final_path = save_path.replace('.png', '_final.png')
        plt.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"   Final results saved to: {final_path}")
    
    plt.show()


def main():
    """
    Main function to demonstrate HDR processing
    """
    parser = argparse.ArgumentParser(description='HDR Image Processing Example')
    parser.add_argument('--input', '-i', type=str, help='Input image path')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    parser.add_argument('--exposures', nargs='+', type=float, 
                       default=[0.25, 0.5, 1.0, 2.0, 4.0],
                       help='Exposure ratios')
    parser.add_argument('--tone-mapping', choices=['reinhard', 'drago'], 
                       default='reinhard', help='Tone mapping method')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Load input image
    if args.input:
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not load image from {args.input}")
            return
    else:
        # Create a sample image if no input provided
        print("📷 Creating sample image...")
        height, width = 400, 600
        
        # Create a synthetic high-contrast scene
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some bright areas (sky)
        image[:height//3, :] = [200, 220, 240]
        
        # Add some mid-tone areas (buildings)
        image[height//3:2*height//3, :] = [100, 80, 60]
        
        # Add some dark areas (foreground)
        image[2*height//3:, :] = [20, 15, 10]
        
        # Add some details
        cv2.circle(image, (width//2, height//2), 50, (255, 255, 255), -1)  # Bright circle
        cv2.rectangle(image, (50, height-100), (150, height-50), (50, 50, 50), -1)  # Dark rectangle
        
        print("   Sample image created")
    
    print(f"📊 Input image shape: {image.shape}")
    
    # Initialize HDR processor
    processor = HDRProcessor()
    
    # Process HDR pipeline
    results = processor.process_hdr_pipeline(
        image, 
        exposure_ratios=args.exposures,
        tone_mapping_method=args.tone_mapping
    )
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        
        # Save tone-mapped result
        output_path = output_dir / 'hdr_result.jpg'
        cv2.imwrite(str(output_path), results['tone_mapped'])
        print(f"💾 HDR result saved to: {output_path}")
        
        # Save HDR image (as EXR or HDR format)
        hdr_path = output_dir / 'hdr_image.exr'
        cv2.imwrite(str(hdr_path), results['hdr_image'])
        print(f"💾 HDR image saved to: {hdr_path}")
        
        # Create and save visualization
        viz_path = output_dir / 'hdr_visualization.png'
        visualize_results(results, str(viz_path))
    
    else:
        # Just show visualization
        visualize_results(results)


if __name__ == "__main__":
    main()
