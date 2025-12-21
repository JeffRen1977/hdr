#!/usr/bin/env python3
"""
HDR (High Dynamic Range) Image Processing Example
=================================================

This is the main entry point for the HDR processing pipeline.
Based on Chapter 13 of "Mobile Computational Photography".

The HDR pipeline demonstrates:
1. Multi-frame exposure capture simulation
2. Image alignment using feature matching
3. HDR fusion using weighted averaging
4. Tone mapping for display

Usage:
    python hdr_example.py --input image.jpg --output results/

Or use the modular API:
    from hdr import HDRProcessor
    from utils import visualize_results
    
    processor = HDRProcessor()
    results = processor.process_hdr_pipeline(image)
    visualize_results(results)
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

from hdr import HDRProcessor
from utils import visualize_results, save_hdr_image


def create_sample_image():
    """Create a sample high-contrast image for demonstration."""
    print("📷 Creating sample image...")
    height, width = 400, 600
    
    # Create a synthetic high-contrast scene
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Bright sky (top third)
    image[:height//3, :] = [200, 220, 240]
    
    # Mid-tone buildings (middle third)
    image[height//3:2*height//3, :] = [100, 80, 60]
    
    # Dark foreground (bottom third)
    image[2*height//3:, :] = [20, 15, 10]
    
    # Add some details for better visualization
    cv2.circle(image, (width//2, height//2), 50, (255, 255, 255), -1)  # Bright circle
    cv2.rectangle(image, (50, height-100), (150, height-50), (50, 50, 50), -1)  # Dark rectangle
    
    print("   Sample image created")
    return image


def main():
    """
    Main function to demonstrate HDR processing.
    
    Command-line interface for the HDR processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description='HDR Image Processing - Based on Chapter 13 of Mobile Computational Photography',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings (creates sample image)
  python hdr_example.py
  
  # Process your own image
  python hdr_example.py --input photo.jpg --output results/
  
  # Custom exposure ratios and tone mapping
  python hdr_example.py -i photo.jpg -o results/ --exposures 0.125 0.25 0.5 1.0 2.0 4.0 8.0 --tone-mapping drago
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input image path (if not provided, creates sample image)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results'
    )
    parser.add_argument(
        '--exposures',
        nargs='+',
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help='Exposure time ratios (default: 0.25 0.5 1.0 2.0 4.0)'
    )
    parser.add_argument(
        '--tone-mapping',
        choices=['reinhard', 'drago'],
        default='reinhard',
        help='Tone mapping method (default: reinhard)'
    )
    parser.add_argument(
        '--key-value',
        type=float,
        default=0.18,
        help='Key value for Reinhard tone mapping (default: 0.18)'
    )
    parser.add_argument(
        '--bias',
        type=float,
        default=0.85,
        help='Bias for Drago tone mapping (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if specified
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
    
    # Load or create input image
    if args.input:
        image = cv2.imread(args.input)
        if image is None:
            print(f"❌ Error: Could not load image from {args.input}")
            return
        print(f"📷 Loaded image: {args.input}")
    else:
        image = create_sample_image()
        print("📷 Using sample image")
    
    print(f"📊 Image shape: {image.shape}")
    
    # Initialize HDR processor
    processor = HDRProcessor()
    
    # Prepare tone mapping parameters
    tone_mapping_kwargs = {}
    if args.tone_mapping == 'reinhard':
        tone_mapping_kwargs['key_value'] = args.key_value
    elif args.tone_mapping == 'drago':
        tone_mapping_kwargs['bias'] = args.bias
    
    # Process HDR pipeline
    results = processor.process_hdr_pipeline(
        image,
        exposure_ratios=args.exposures,
        tone_mapping_method=args.tone_mapping,
        **tone_mapping_kwargs
    )
    
    # Save results
    if output_dir:
        # Save tone-mapped result
        output_path = output_dir / 'hdr_result.jpg'
        cv2.imwrite(str(output_path), results['tone_mapped'])
        print(f"💾 HDR result saved to: {output_path}")
        
        # Save HDR image (with automatic format fallback)
        hdr_path = output_dir / 'hdr_image.exr'
        save_hdr_image(results['hdr_image'], hdr_path)
        
        # Create and save visualization
        viz_path = output_dir / 'hdr_visualization.png'
        visualize_results(results, str(viz_path))
    else:
        # Just display visualization
        visualize_results(results)
    
    print("\n🎉 Processing complete!")


if __name__ == "__main__":
    main()