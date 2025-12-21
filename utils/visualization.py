"""
Visualization Utilities
=======================

Functions for visualizing HDR processing pipeline results.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional


def visualize_results(results: dict, save_path: Optional[str] = None):
    """
    Create comprehensive visualization of HDR processing results.
    
    Shows:
    1. Input exposure images (first 3)
    2. Weight maps (first 3)
    3. HDR image (log scale for better visibility)
    4. Final tone-mapped result
    
    Args:
        results: Dictionary from HDRProcessor.process_hdr_pipeline containing:
                - 'exposure_images': List of exposure images
                - 'exposure_times': List of exposure time ratios
                - 'weights': List of weight maps
                - 'hdr_image': HDR image (float32, 0-1 range)
                - 'tone_mapped': Tone-mapped LDR image (uint8)
        save_path: Optional path to save visualization (if None, just display)
    """
    print("📊 Creating visualization...")
    
    # Create figure showing exposure images and weight maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('HDR Processing Pipeline - Exposures and Weights', fontsize=16)
    
    # Show first 3 exposure images
    for i in range(min(3, len(results['exposure_images']))):
        img = results['exposure_images'][i]
        # Convert BGR to RGB for matplotlib
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        axes[0, i].imshow(img_rgb, cmap='gray' if len(img.shape) == 2 else None)
        axes[0, i].set_title(f'Exposure {results["exposure_times"][i]:.2f}x')
        axes[0, i].axis('off')
    
    # Show first 3 weight maps
    for i in range(min(3, len(results['weights']))):
        weight = results['weights'][i]
        # Use first channel for grayscale display if color
        if len(weight.shape) == 3:
            weight = weight[:, :, 0]
        
        axes[1, i].imshow(weight, cmap='hot')
        axes[1, i].set_title(f'Weight Map {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved to: {save_path}")
    
    plt.show()
    
    # Create figure showing HDR and final result
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('HDR Processing Results', fontsize=16)
    
    # HDR image (use log scale for better visualization since HDR has high dynamic range)
    hdr_image = results['hdr_image']
    hdr_log = np.log(hdr_image + 1e-8)  # Add small epsilon to avoid log(0)
    hdr_log_normalized = (hdr_log - hdr_log.min()) / (hdr_log.max() - hdr_log.min())
    
    if len(hdr_image.shape) == 3:
        hdr_rgb = cv2.cvtColor((hdr_log_normalized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        axes[0].imshow(hdr_rgb)
    else:
        axes[0].imshow(hdr_log_normalized, cmap='gray')
    axes[0].set_title('HDR Image (Log Scale)')
    axes[0].axis('off')
    
    # Tone-mapped final result
    tone_mapped = results['tone_mapped']
    if len(tone_mapped.shape) == 3:
        tone_mapped_rgb = cv2.cvtColor(tone_mapped, cv2.COLOR_BGR2RGB)
        axes[1].imshow(tone_mapped_rgb)
    else:
        axes[1].imshow(tone_mapped, cmap='gray')
    axes[1].set_title('Tone-Mapped Result (Final)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        final_path = save_path.replace('.png', '_final.png')
        plt.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f"   Final results saved to: {final_path}")
    
    plt.show()
