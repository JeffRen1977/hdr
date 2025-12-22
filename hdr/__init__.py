"""
HDR (High Dynamic Range) Image Processing Package
==================================================

This package implements the HDR processing pipeline based on Chapter 13 of 
"Mobile Computational Photography" textbook.

The HDR pipeline consists of four main stages:
1. Multi-frame exposure capture (simulation)
2. Image alignment using feature matching
3. HDR fusion using weighted averaging
4. Tone mapping for display

Main Components:
    - processor: Main HDRProcessor class that orchestrates the pipeline
    - alignment: Image alignment algorithms
    - fusion: HDR fusion and weight computation
    - tone_mapping: Tone mapping algorithms (Reinhard, Drago)

Usage:
    from hdr import HDRProcessor
    
    processor = HDRProcessor()
    results = processor.process_hdr_pipeline(image)
"""

from hdr.processor import HDRProcessor

__all__ = ['HDRProcessor']
__version__ = '1.0.0'

