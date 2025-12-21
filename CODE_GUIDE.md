# Code Guide: Understanding the HDR Implementation

This guide helps you navigate the codebase and understand how each component relates to Chapter 13 of "Mobile Computational Photography".

## Quick Start

**Want to run HDR processing?**
```bash
python hdr_example.py --input your_image.jpg --output results/
```

**Want to understand how it works?** Start here:
1. Read `hdr/processor.py` - See how the pipeline is orchestrated
2. Read individual modules - Understand each component
3. Check `README.md` - Comprehensive documentation

## Code Organization

### 📁 `hdr/` - Core HDR Processing Modules

#### `processor.py` - Main Orchestrator ⭐ START HERE
- **Purpose**: Coordinates the entire HDR pipeline
- **Key Class**: `HDRProcessor`
- **What it does**: 
  - Takes a single input image
  - Simulates multiple exposures
  - Calls alignment, fusion, and tone mapping modules
  - Returns complete results

**Reading Order**: Start here to understand the overall flow.

#### `alignment.py` - Image Registration
- **Purpose**: Aligns multiple exposure images to prevent ghosting
- **Key Class**: `ImageAligner`
- **Algorithm**: ORB feature detection + FLANN matching + Homography
- **Chapter 13 Concept**: Addresses camera shake and object motion challenges

**When to read**: When you want to understand how images are aligned.

#### `fusion.py` - HDR Image Fusion
- **Purpose**: Combines multiple exposures into a single HDR image
- **Key Class**: `HDRFusion`
- **Algorithm**: Weighted averaging with triangle weighting function
- **Chapter 13 Concept**: Multi-frame compositing with intelligent weighting

**When to read**: When you want to understand how exposures are combined.

#### `tone_mapping.py` - HDR to LDR Conversion
- **Purpose**: Converts HDR image to displayable format
- **Key Class**: `ToneMapper`
- **Algorithms**: Reinhard and Drago tone mapping
- **Chapter 13 Concept**: Post-processing optimization for display

**When to read**: When you want to understand tone mapping algorithms.

### 📁 `utils/` - Helper Functions

#### `image_utils.py` - Image Processing Utilities
- **Purpose**: Helper functions for image manipulation
- **Key Function**: `simulate_multi_exposure_capture()`
- **What it does**: Simulates different exposure times by scaling pixel values

#### `visualization.py` - Result Visualization
- **Purpose**: Visualize HDR processing results
- **Key Function**: `visualize_results()`
- **What it does**: Creates plots showing exposures, weights, HDR, and final result

### 📄 Root Files

#### `hdr_example.py` - Main Entry Point
- **Purpose**: Command-line interface for HDR processing
- **What it does**: Parses arguments, loads images, runs pipeline, saves results
- **Usage**: `python hdr_example.py --input image.jpg --output results/`

#### `run_hdr.py` - Advanced Runner
- **Purpose**: More advanced runner with multiple scenarios
- **What it does**: Runs basic, advanced, and comparison tests
- **Usage**: `python run_hdr.py` or `./run.sh`

## Reading Path Recommendations

### Path 1: Understand the Complete Pipeline (Recommended for Beginners)

1. **`hdr/processor.py`** - Read `process_hdr_pipeline()` method
   - See how all components work together
   - Understand the 5-stage pipeline

2. **`utils/image_utils.py`** - Read `simulate_multi_exposure_capture()`
   - Understand how exposures are simulated

3. **`hdr/alignment.py`** - Read `align_images()` method
   - Understand feature-based alignment

4. **`hdr/fusion.py`** - Read both `compute_weights()` and `fuse_hdr()`
   - Understand weighting and fusion

5. **`hdr/tone_mapping.py`** - Read tone mapping methods
   - Understand HDR to LDR conversion

### Path 2: Focus on Specific Algorithms

**Want to understand alignment?**
→ Read `hdr/alignment.py` from top to bottom

**Want to understand fusion?**
→ Read `hdr/fusion.py` from top to bottom

**Want to understand tone mapping?**
→ Read `hdr/tone_mapping.py` from top to bottom

### Path 3: Follow Data Flow

1. Input image → `hdr_example.py` (loads image)
2. → `HDRProcessor.process_hdr_pipeline()` (orchestrates)
3. → `simulate_multi_exposure_capture()` (creates exposures)
4. → `ImageAligner.align_images()` (aligns exposures)
5. → `HDRFusion.compute_weights()` (computes weights)
6. → `HDRFusion.fuse_hdr()` (fuses to HDR)
7. → `ToneMapper.apply_tone_mapping()` (converts to LDR)
8. → Output saved/displayed

## Key Concepts from Chapter 13

Each module corresponds to concepts from Chapter 13:

| Module | Chapter 13 Concept | Implementation |
|--------|-------------------|----------------|
| `image_utils.py` | Multi-frame exposure capture | `simulate_multi_exposure_capture()` |
| `alignment.py` | Image alignment for ghosting prevention | `ImageAligner.align_images()` |
| `fusion.py` | Dynamic range fusion | `HDRFusion.fuse_hdr()` |
| `tone_mapping.py` | Post-processing optimization | `ToneMapper.apply_tone_mapping()` |

## Code Quality Features

✅ **Clear Separation of Concerns**: Each module has a single responsibility  
✅ **Comprehensive Documentation**: Every function has detailed docstrings  
✅ **Type Hints**: Function signatures include type information  
✅ **Educational Comments**: Code explains *why*, not just *what*  
✅ **Modular Design**: Components can be used independently  

## Extension Points

Want to add your own algorithm? Here's where to modify:

- **New alignment algorithm**: Add method to `ImageAligner` class
- **New fusion algorithm**: Add method to `HDRFusion` class  
- **New tone mapping**: Add method to `ToneMapper` class
- **New utility function**: Add to appropriate `utils/` module

## Setting Up and Testing the Code

### Virtual Environment Setup

The project uses a virtual environment located one directory above the HDR folder:

```bash
# Directory structure:
# Mobile_Computational_photograph/
#   ├── virtual_ven/          # Virtual environment
#   └── HDR/                  # Project directory

# Activate virtual environment (from HDR directory):
cd ..
source virtual_ven/bin/activate  # On macOS/Linux
cd HDR

# Or activate from parent directory:
# cd Mobile_Computational_photograph/
# source virtual_ven/bin/activate
# cd HDR
```

### Testing the Code

```bash
# Make sure virtual environment is activated first!

# Test imports
python3 -c "from hdr import HDRProcessor; print('OK')"

# Run with sample image
python hdr_example.py

# Run advanced tests
python run_hdr.py

# Or use the bash script
./run.sh
```

## Questions?

- **How does alignment work?** → See `hdr/alignment.py`
- **How are weights computed?** → See `hdr/fusion.py` → `compute_weights()`
- **How is fusion done?** → See `hdr/fusion.py` → `fuse_hdr()`
- **What is tone mapping?** → See `hdr/tone_mapping.py`
- **How to use the code?** → See `README.md` Usage section

---

Happy coding! 🎉
