#!/usr/bin/env python3
"""
HDR Runner Script
=================

Simple script to run the HDR example with different configurations.
This script demonstrates various HDR processing scenarios.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from hdr_example import HDRProcessor, visualize_results


def create_sample_images():
    """Create sample images for demonstration if no input provided"""
    print("🎨 Creating sample images for demonstration...")
    
    # Create output directory for samples
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a high-contrast sample scene
    height, width = 600, 800
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky gradient (bright to medium)
    for y in range(height//3):
        intensity = int(255 - (y / (height//3)) * 100)
        image[y, :] = [intensity, intensity + 20, intensity + 40]
    
    # Building silhouette (dark)
    image[height//3:2*height//3, :] = [30, 25, 20]
    
    # Ground (medium dark)
    image[2*height//3:, :] = [60, 50, 40]
    
    # Add some bright elements (sun, windows)
    cv2.circle(image, (width//2, height//6), 40, (255, 255, 200), -1)  # Sun
    cv2.rectangle(image, (100, height//3 + 50), (200, height//3 + 150), (255, 255, 255), -1)  # Window
    cv2.rectangle(image, (600, height//3 + 30), (700, height//3 + 130), (255, 255, 255), -1)  # Window
    
    # Add some dark elements (trees, shadows)
    cv2.ellipse(image, (150, height - 100), (30, 80), 0, 0, 360, (10, 20, 5), -1)  # Tree
    cv2.ellipse(image, (650, height - 120), (25, 70), 0, 0, 360, (8, 15, 3), -1)   # Tree
    
    # Save sample image
    sample_path = sample_dir / "sample_scene.jpg"
    cv2.imwrite(str(sample_path), image)
    print(f"   Sample image saved to: {sample_path}")
    
    return str(sample_path)


def run_basic_hdr(input_path=None, output_dir="hdr_results"):
    """Run basic HDR processing"""
    print("\n" + "="*60)
    print("🚀 RUNNING BASIC HDR PROCESSING")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load or create input image
    if input_path and os.path.exists(input_path):
        image = cv2.imread(input_path)
        print(f"📷 Loaded image: {input_path}")
    else:
        input_path = create_sample_images()
        image = cv2.imread(input_path)
        print(f"📷 Created sample image: {input_path}")
    
    if image is None:
        print("❌ Error: Could not load image")
        return False
    
    print(f"📊 Image shape: {image.shape}")
    
    # Initialize HDR processor
    processor = HDRProcessor()
    
    # Process HDR with default settings
    print("\n🔧 Processing with default settings...")
    results = processor.process_hdr_pipeline(image)
    
    # Save results
    output_path = Path(output_dir) / "basic_hdr_result.jpg"
    cv2.imwrite(str(output_path), results['tone_mapped'])
    print(f"💾 Basic HDR result saved to: {output_path}")
    
    # Create visualization
    viz_path = Path(output_dir) / "basic_hdr_visualization.png"
    visualize_results(results, str(viz_path))
    
    return True


def run_advanced_hdr(input_path=None, output_dir="hdr_results"):
    """Run advanced HDR processing with custom settings"""
    print("\n" + "="*60)
    print("🚀 RUNNING ADVANCED HDR PROCESSING")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load or create input image
    if input_path and os.path.exists(input_path):
        image = cv2.imread(input_path)
        print(f"📷 Loaded image: {input_path}")
    else:
        input_path = create_sample_images()
        image = cv2.imread(input_path)
        print(f"📷 Created sample image: {input_path}")
    
    if image is None:
        print("❌ Error: Could not load image")
        return False
    
    # Initialize HDR processor
    processor = HDRProcessor()
    
    # Advanced settings
    exposure_ratios = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # More exposures
    tone_mapping_method = 'drago'  # Use Drago tone mapping
    
    print(f"🔧 Processing with advanced settings:")
    print(f"   - Exposure ratios: {exposure_ratios}")
    print(f"   - Tone mapping: {tone_mapping_method}")
    
    results = processor.process_hdr_pipeline(
        image, 
        exposure_ratios=exposure_ratios,
        tone_mapping_method=tone_mapping_method
    )
    
    # Save results
    output_path = Path(output_dir) / "advanced_hdr_result.jpg"
    cv2.imwrite(str(output_path), results['tone_mapped'])
    print(f"💾 Advanced HDR result saved to: {output_path}")
    
    # Save HDR image
    hdr_path = Path(output_dir) / "advanced_hdr_image.exr"
    cv2.imwrite(str(hdr_path), results['hdr_image'])
    print(f"💾 HDR image saved to: {hdr_path}")
    
    # Create visualization
    viz_path = Path(output_dir) / "advanced_hdr_visualization.png"
    visualize_results(results, str(viz_path))
    
    return True


def run_comparison_test(input_path=None, output_dir="hdr_results"):
    """Run comparison between different tone mapping methods"""
    print("\n" + "="*60)
    print("🚀 RUNNING TONE MAPPING COMPARISON")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load or create input image
    if input_path and os.path.exists(input_path):
        image = cv2.imread(input_path)
        print(f"📷 Loaded image: {input_path}")
    else:
        input_path = create_sample_images()
        image = cv2.imread(input_path)
        print(f"📷 Created sample image: {input_path}")
    
    if image is None:
        print("❌ Error: Could not load image")
        return False
    
    # Initialize HDR processor
    processor = HDRProcessor()
    
    # Test both tone mapping methods
    tone_mapping_methods = ['reinhard', 'drago']
    
    for method in tone_mapping_methods:
        print(f"\n🔧 Testing {method.upper()} tone mapping...")
        
        results = processor.process_hdr_pipeline(
            image, 
            exposure_ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            tone_mapping_method=method
        )
        
        # Save results
        output_path = Path(output_dir) / f"hdr_{method}_result.jpg"
        cv2.imwrite(str(output_path), results['tone_mapped'])
        print(f"💾 {method.title()} result saved to: {output_path}")
        
        # Create visualization
        viz_path = Path(output_dir) / f"hdr_{method}_visualization.png"
        visualize_results(results, str(viz_path))
    
    return True


def main():
    """Main function to run HDR demonstrations"""
    print("🎯 HDR Processing Runner")
    print("=" * 60)
    print("This script demonstrates the HDR processing pipeline")
    print("with different configurations and settings.")
    print("=" * 60)
    
    # Check if input image is provided as command line argument
    input_path = None
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not os.path.exists(input_path):
            print(f"⚠️  Warning: Input file '{input_path}' not found, will create sample image")
            input_path = None
    
    # Check if output directory is provided as command line argument
    output_dir = "hdr_results"
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Run different HDR processing scenarios
        success_count = 0
        
        # 1. Basic HDR processing
        if run_basic_hdr(input_path, output_dir):
            success_count += 1
        
        # 2. Advanced HDR processing
        if run_advanced_hdr(input_path, output_dir):
            success_count += 1
        
        # 3. Tone mapping comparison
        if run_comparison_test(input_path, output_dir):
            success_count += 1
        
        # Summary
        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        print(f"✅ Successfully completed {success_count}/3 processing scenarios")
        print(f"📁 All results saved to: {output_dir}/")
        print("\nGenerated files:")
        
        output_path = Path(output_dir)
        if output_path.exists():
            for file in sorted(output_path.glob("*")):
                print(f"   - {file.name}")
        
        print("\n🎉 HDR processing completed successfully!")
        print("💡 Tip: Check the visualization images to see the pipeline steps")
        
    except Exception as e:
        print(f"\n❌ Error during HDR processing: {e}")
        print("💡 Make sure you have all required dependencies installed:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
