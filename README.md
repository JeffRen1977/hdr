# Mobile Computational Photography: HDR Implementation

This repository demonstrates a complete High Dynamic Range (HDR) image processing pipeline, implementing the core concepts outlined in the mobile computational photography design document.

## Table of Contents

1. [Overview](#overview)
2. [HDR Technology Fundamentals](#hdr-technology-fundamentals)
3. [Implementation Architecture](#implementation-architecture)
4. [HDR Processing Pipeline](#hdr-processing-pipeline)
5. [Code Example Walkthrough](#code-example-walkthrough)
6. [Usage Instructions](#usage-instructions)
7. [Technical Deep Dive](#technical-deep-dive)
8. [Future Enhancements](#future-enhancements)
9. [References](#references)

---

## Overview

High Dynamic Range (HDR) imaging is a computational photography technique that captures and displays a wider range of luminance levels than conventional digital imaging. This implementation demonstrates the complete HDR workflow from multi-frame capture simulation to final tone-mapped output, following the principles outlined in our mobile computational photography framework.

### Key Features

- **Multi-Frame Exposure Simulation**: Simulates different exposure times to capture varying dynamic ranges
- **Advanced Image Alignment**: Feature-based registration using ORB detectors and FLANN matching
- **Intelligent Fusion**: Weighted averaging based on pixel value distributions
- **Multiple Tone Mapping**: Reinhard and Drago tone mapping algorithms
- **Comprehensive Visualization**: Step-by-step pipeline visualization
- **Real-World Ready**: Designed for mobile computational photography applications

---

## HDR Technology Fundamentals

### What is HDR?

HDR (High Dynamic Range) technology aims to capture and present a broader range of light intensities that more closely matches human visual perception. Traditional cameras and displays have limited dynamic range, often losing details in very bright (overexposed) or very dark (underexposed) areas.

### The HDR Challenge

The human eye can perceive a dynamic range of approximately 10^4 to 10^5, while typical digital cameras capture only 10^2 to 10^3. This limitation results in:

- **Lost highlights**: Bright areas become pure white with no detail
- **Lost shadows**: Dark areas become pure black with no detail
- **Compressed mid-tones**: Reduced contrast in normal exposure ranges

### HDR Solution Strategy

HDR addresses these limitations through a multi-step process:

1. **Capture Multiple Exposures**: Take several images at different exposure settings
2. **Align Images**: Compensate for camera movement and object motion
3. **Fuse Information**: Combine the best parts of each exposure
4. **Tone Map**: Compress the high dynamic range for standard displays

---

## Implementation Architecture

Our HDR implementation follows a modular architecture that mirrors the mobile computational photography pipeline:

```
Input Image
    ↓
Multi-Exposure Simulation
    ↓
Image Alignment (Feature Matching)
    ↓
Weight Computation
    ↓
HDR Fusion
    ↓
Tone Mapping
    ↓
Final Output
```

### Core Components

#### 1. HDRProcessor Class
The main processing engine that orchestrates the entire pipeline:

```python
class HDRProcessor:
    def __init__(self):
        self.aligned_images: List[np.ndarray] = []
        self.exposure_times: List[float] = []
        self.hdr_image: Optional[np.ndarray] = None
```

#### 2. Multi-Exposure Capture Simulation
Simulates the process of capturing multiple exposures by applying different exposure adjustments to a base image.

#### 3. Image Alignment Engine
Uses computer vision techniques to align multiple exposures, compensating for camera shake and object movement.

#### 4. Fusion Algorithm
Implements weighted averaging based on pixel value distributions to combine information from multiple exposures.

#### 5. Tone Mapping System
Converts high dynamic range data to displayable formats using proven algorithms.

---

## HDR Processing Pipeline

### Step 1: Multi-Frame Exposure Capture

The first step simulates capturing multiple images at different exposure settings. This is achieved by:

```python
def simulate_multi_exposure_capture(self, base_image: np.ndarray, 
                                  exposure_ratios: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0]):
    """
    Simulate different exposure times by scaling pixel values
    Higher exposure = brighter image (multiply by ratio)
    Lower exposure = darker image (divide by ratio)
    """
```

**Technical Details:**
- **Exposure Ratios**: Typically [0.25, 0.5, 1.0, 2.0, 4.0] covering ±2 stops
- **Pixel Scaling**: Linear scaling based on exposure time ratios
- **Clipping**: Prevents overflow beyond valid pixel ranges

### Step 2: Image Alignment

Image alignment is crucial for HDR quality. Even slight camera movement between exposures can cause ghosting and artifacts.

```python
def align_images(self, images: List[np.ndarray], reference_idx: int = 2):
    """
    Align multiple exposure images using feature-based registration
    Uses ORB detector for feature matching and homography estimation
    """
```

**Technical Implementation:**
- **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF) detector
- **Feature Matching**: FLANN-based matcher with ratio test filtering
- **Transformation**: Homography estimation using RANSAC
- **Warping**: Perspective transformation for alignment

### Step 3: Weight Computation

The fusion process uses intelligent weighting to emphasize well-exposed pixels:

```python
def compute_weights(self, images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Triangle weighting function that gives higher weight to 
    pixels that are not too dark or too bright
    """
```

**Weighting Strategy:**
- **Triangle Function**: Peak weight at 50% gray value
- **Linear Decay**: Weight decreases toward extremes (0% and 100%)
- **Channel Averaging**: For color images, weights are computed per channel

### Step 4: HDR Fusion

The core fusion algorithm combines aligned exposures using computed weights:

```python
def fuse_hdr(self, aligned_images: List[np.ndarray], 
            exposure_times: List[float], weights: List[np.ndarray]):
    """
    Fuse aligned exposure images into HDR image using weighted averaging
    """
```

**Fusion Process:**
1. **Exposure Compensation**: Normalize pixel values by exposure time
2. **Weighted Accumulation**: Sum weighted, compensated images
3. **Normalization**: Divide by total weight sum
4. **Range Clipping**: Ensure valid output range

### Step 5: Tone Mapping

Tone mapping converts HDR data to displayable LDR (Low Dynamic Range) format:

```python
def tone_map_reinhard(self, hdr_image: np.ndarray, key_value: float = 0.18):
    """
    Apply Reinhard tone mapping to convert HDR to LDR
    """
```

**Available Methods:**
- **Reinhard**: Physically-based, preserves local contrast
- **Drago**: Optimized for display devices, better color preservation

---

## Code Example Walkthrough

### Basic Usage

The simplest way to use our HDR implementation:

```python
import cv2
from hdr_example import HDRProcessor

# Load an image
image = cv2.imread('sample.jpg')

# Create processor
processor = HDRProcessor()

# Process HDR pipeline
results = processor.process_hdr_pipeline(image)

# Access results
hdr_image = results['hdr_image']
tone_mapped = results['tone_mapped']
```

### Advanced Configuration

For more control over the HDR process:

```python
# Custom exposure ratios
exposure_ratios = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# Use Drago tone mapping
results = processor.process_hdr_pipeline(
    image, 
    exposure_ratios=exposure_ratios,
    tone_mapping_method='drago'
)

# Visualize results
from hdr_example import visualize_results
visualize_results(results, save_path='hdr_visualization.png')
```

### Command Line Usage

The implementation includes a command-line interface:

```bash
# Basic usage with default settings
python hdr_example.py

# With custom input image
python hdr_example.py --input photo.jpg --output results/

# Custom exposure ratios
python hdr_example.py --exposures 0.125 0.25 0.5 1.0 2.0 4.0 8.0

# Use Drago tone mapping
python hdr_example.py --tone-mapping drago
```

---

## Usage Instructions

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd HDR
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Example

1. **Basic execution:**
```bash
python hdr_example.py
```

2. **With your own image:**
```bash
python hdr_example.py --input your_image.jpg --output results/
```

3. **Custom configuration:**
```bash
python hdr_example.py \
    --input photo.jpg \
    --output hdr_results/ \
    --exposures 0.25 0.5 1.0 2.0 4.0 \
    --tone-mapping reinhard
```

### Output Files

The implementation generates several output files:

- `hdr_result.jpg`: Final tone-mapped HDR image
- `hdr_image.exr`: Raw HDR data (OpenEXR format)
- `hdr_visualization.png`: Pipeline visualization
- `hdr_visualization_final.png`: Final results comparison

---

## Technical Deep Dive

### Mathematical Foundations

#### Exposure Compensation
The relationship between pixel value and scene radiance is:
```
L(x,y) = I(x,y) / t
```
Where:
- `L(x,y)`: Scene radiance at pixel (x,y)
- `I(x,y)`: Pixel intensity value
- `t`: Exposure time

#### Weight Function
The triangle weighting function is defined as:
```
w(I) = {
    2I,           if I ≤ 0.5
    2(1-I),       if I > 0.5
}
```

#### HDR Fusion
The final HDR image is computed as:
```
HDR(x,y) = Σ[I_i(x,y) * w_i(x,y) / t_i] / Σ[w_i(x,y)]
```

#### Reinhard Tone Mapping
```
L_d = L * (1 + L/L_w²) / (1 + L)
```
Where:
- `L_d`: Display luminance
- `L`: Scene luminance
- `L_w`: World adaptation luminance (key value)

### Performance Considerations

#### Computational Complexity
- **Feature Detection**: O(n²) where n is image dimension
- **Feature Matching**: O(m log m) where m is number of features
- **HDR Fusion**: O(p) where p is total pixels
- **Tone Mapping**: O(p)

#### Memory Usage
- **Input Images**: n_images × height × width × channels
- **HDR Image**: height × width × channels (float32)
- **Weight Maps**: n_images × height × width × channels

#### Optimization Strategies
1. **Feature Limit**: Cap maximum features per image
2. **Pyramid Processing**: Multi-scale alignment for efficiency
3. **GPU Acceleration**: OpenCV CUDA support for large images
4. **Memory Management**: Process images in batches for large datasets

### Quality Metrics

#### Objective Quality Measures
- **Dynamic Range**: Ratio of maximum to minimum luminance
- **Signal-to-Noise Ratio**: Noise level in final image
- **Ghosting Artifacts**: Misalignment detection
- **Color Accuracy**: Color fidelity preservation

#### Subjective Quality Factors
- **Natural Appearance**: Visual realism
- **Detail Preservation**: Fine detail retention
- **Contrast Enhancement**: Local contrast improvement
- **Artifact Minimization**: Reduction of processing artifacts

---

## Future Enhancements

### Deep Learning Integration

Following the mobile computational photography trends outlined in the design document, future versions will integrate:

#### 1. Neural HDR Fusion
```python
# Future enhancement - Neural fusion
class NeuralHDRFusion:
    def __init__(self, model_path):
        self.fusion_net = load_pretrained_model(model_path)
    
    def fuse_hdr(self, aligned_images, exposure_times):
        return self.fusion_net(aligned_images, exposure_times)
```

#### 2. Learned Tone Mapping
```python
# Future enhancement - Learned tone mapping
class LearnedToneMapping:
    def __init__(self, model_path):
        self.tone_mapping_net = load_pretrained_model(model_path)
    
    def tone_map(self, hdr_image):
        return self.tone_mapping_net(hdr_image)
```

#### 3. Motion Deghosting
```python
# Future enhancement - Motion artifact removal
class MotionDeghosting:
    def __init__(self, deghosting_model):
        self.model = deghosting_model
    
    def remove_ghosts(self, aligned_images):
        return self.model(aligned_images)
```

### Mobile Optimization

#### 1. Real-time Processing
- **GPU Acceleration**: CUDA/OpenCL implementation
- **SIMD Optimization**: Vectorized operations
- **Memory Pooling**: Reduced memory allocations
- **Pipeline Parallelism**: Concurrent processing stages

#### 2. Power Efficiency
- **Adaptive Quality**: Dynamic quality adjustment based on battery
- **Sensor Fusion**: Integration with IMU data for better alignment
- **Hardware Acceleration**: NPU utilization for neural components

#### 3. Camera Integration
- **RAW Processing**: Direct RAW sensor data processing
- **3A Integration**: Automatic exposure, focus, and white balance
- **Burst Mode**: Optimized for continuous capture

### Advanced Features

#### 1. Video HDR
```python
# Future enhancement - Video HDR
class VideoHDRProcessor:
    def process_video_hdr(self, video_stream):
        # Temporal consistency
        # Real-time processing
        # Memory-efficient buffering
        pass
```

#### 2. Semantic-Aware HDR
```python
# Future enhancement - Scene understanding
class SemanticHDR:
    def __init__(self, segmentation_model):
        self.segmentation = segmentation_model
    
    def adaptive_fusion(self, images, scene_segments):
        # Different fusion strategies for different scene regions
        pass
```

#### 3. Computational Photography Pipeline
```python
# Future enhancement - Complete pipeline
class ComputationalPhotographyPipeline:
    def __init__(self):
        self.hdr_processor = HDRProcessor()
        self.denoising_net = DenoisingNetwork()
        self.enhancement_net = EnhancementNetwork()
    
    def process_complete(self, raw_data):
        # Complete computational photography workflow
        pass
```

---

## References

### Academic Papers
1. **Reinhard, E., et al.** "Photographic tone reproduction for digital images." ACM Transactions on Graphics 21.3 (2002): 267-276.
2. **Drago, F., et al.** "Adaptive logarithmic mapping for displaying high contrast scenes." Computer Graphics Forum 22.3 (2003): 419-426.
3. **Debevec, P. E., and J. Malik.** "Recovering high dynamic range radiance maps from photographs." Proceedings of the 24th annual conference on Computer graphics and interactive techniques. 1997.

### Technical Resources
1. **OpenCV Documentation**: Computer vision algorithms and implementations
2. **OpenEXR Specification**: High dynamic range image format
3. **IEEE Computer Graphics**: HDR imaging standards and techniques

### Mobile Computational Photography
1. **Apple Computational Photography**: iPhone HDR implementation insights
2. **Google HDR+**: Mobile HDR processing pipeline
3. **Samsung Multi-Frame Processing**: Galaxy HDR techniques

### Open Source Projects
1. **OpenHDR**: Open source HDR processing library
2. **Luminance HDR**: Cross-platform HDR software
3. **pfstools**: HDR image processing toolkit

---

## Contributing

We welcome contributions to improve this HDR implementation:

1. **Algorithm Improvements**: Better fusion or tone mapping algorithms
2. **Performance Optimization**: Speed and memory usage improvements
3. **Mobile Integration**: Platform-specific optimizations
4. **Deep Learning**: Neural network-based enhancements
5. **Documentation**: Improved explanations and examples

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest jupyter

# Run tests
pytest tests/

# Interactive development
jupyter notebook
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **Mobile Computational Photography Community**: For advancing HDR research
- **OpenCV Contributors**: For providing robust computer vision tools
- **Academic Researchers**: For foundational HDR algorithms and techniques

---

*This implementation demonstrates the practical application of mobile computational photography principles, providing a solid foundation for understanding and implementing HDR technology in real-world applications.*
