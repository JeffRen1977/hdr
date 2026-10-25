"""
Tests for the HDR processing pipeline.

Tests cover the core modules: tone_mapping, fusion, alignment, and processor.
Uses synthetic images so no external data files are needed.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdr.tone_mapping import ToneMapper
from hdr.fusion import HDRFusion
from hdr.alignment import ImageAligner
from hdr.processor import HDRProcessor
from utils.image_utils import simulate_multi_exposure_capture


def make_synthetic_image(height=64, width=64):
    """Create a synthetic test image with a gradient pattern."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        val = int(255 * i / height)
        img[i, :, :] = val
    return img


class TestToneMapping:
    def test_reinhard_output_shape(self):
        hdr_img = np.random.rand(32, 32, 3).astype(np.float32)
        result = ToneMapper.tone_map_reinhard(hdr_img)
        assert result.shape == hdr_img.shape
        assert result.dtype == np.uint8

    def test_reinhard_output_range(self):
        hdr_img = np.random.rand(32, 32, 3).astype(np.float32)
        result = ToneMapper.tone_map_reinhard(hdr_img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_drago_output_shape(self):
        hdr_img = np.random.rand(32, 32, 3).astype(np.float32)
        result = ToneMapper.tone_map_drago(hdr_img)
        assert result.shape == hdr_img.shape
        assert result.dtype == np.uint8

    def test_apply_tone_mapping_reinhard(self):
        hdr_img = np.random.rand(32, 32, 3).astype(np.float32)
        result = ToneMapper.apply_tone_mapping(hdr_img, method='reinhard')
        assert result.dtype == np.uint8

    def test_apply_tone_mapping_drago(self):
        hdr_img = np.random.rand(32, 32, 3).astype(np.float32)
        result = ToneMapper.apply_tone_mapping(hdr_img, method='drago')
        assert result.dtype == np.uint8

    def test_black_image(self):
        hdr_img = np.zeros((32, 32, 3), dtype=np.float32)
        result = ToneMapper.tone_map_reinhard(hdr_img)
        assert result.max() == 0

    def test_grayscale_image(self):
        hdr_img = np.random.rand(32, 32).astype(np.float32)
        result = ToneMapper.tone_map_reinhard(hdr_img)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8


class TestHDRFusion:
    def test_compute_weights_count(self):
        images = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
        weights = HDRFusion.compute_weights(images)
        assert len(weights) == 3

    def test_compute_weights_shape(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        weights = HDRFusion.compute_weights([img])
        assert weights[0].shape == img.shape

    def test_compute_weights_range(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        weights = HDRFusion.compute_weights([img])
        assert weights[0].min() >= 0.0
        assert weights[0].max() <= 1.0

    def test_midtone_gets_highest_weight(self):
        dark = np.zeros((4, 4, 3), dtype=np.uint8)
        mid = np.full((4, 4, 3), 128, dtype=np.uint8)
        bright = np.full((4, 4, 3), 255, dtype=np.uint8)
        weights = HDRFusion.compute_weights([dark, mid, bright])
        assert weights[1].mean() > weights[0].mean()
        assert weights[1].mean() > weights[2].mean()

    def test_fuse_hdr_output_shape(self):
        images = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
        times = [0.5, 1.0, 2.0]
        weights = HDRFusion.compute_weights(images)
        hdr = HDRFusion.fuse_hdr(images, times, weights)
        assert hdr.shape == images[0].shape
        assert hdr.dtype == np.float32

    def test_fuse_hdr_output_range(self):
        images = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
        times = [0.5, 1.0, 2.0]
        weights = HDRFusion.compute_weights(images)
        hdr = HDRFusion.fuse_hdr(images, times, weights)
        assert hdr.min() >= 0.0
        assert hdr.max() <= 1.0


class TestImageAligner:
    def test_init(self):
        aligner = ImageAligner(n_features=500, match_ratio=0.8)
        assert aligner.n_features == 500
        assert aligner.match_ratio == 0.8

    def test_single_image_passthrough(self):
        img = make_synthetic_image()
        aligner = ImageAligner()
        result = aligner.align_images([img])
        assert len(result) == 1

    def test_identical_images_alignment(self):
        img = make_synthetic_image(128, 128)
        images = [img.copy() for _ in range(3)]
        aligner = ImageAligner()
        aligned = aligner.align_images(images, reference_idx=1)
        assert len(aligned) >= 1


class TestSimulateExposure:
    def test_output_count(self):
        img = make_synthetic_image()
        ratios = [0.5, 1.0, 2.0]
        exposures, times = simulate_multi_exposure_capture(img, ratios)
        assert len(exposures) == len(ratios)
        assert len(times) == len(ratios)

    def test_output_shape(self):
        img = make_synthetic_image()
        exposures, times = simulate_multi_exposure_capture(img, [0.5, 1.0, 2.0])
        for exp in exposures:
            assert exp.shape == img.shape
            assert exp.dtype == np.uint8

    def test_exposure_ordering(self):
        img = make_synthetic_image()
        exposures, times = simulate_multi_exposure_capture(img, [0.25, 1.0, 4.0])
        mean_brightness = [exp.mean() for exp in exposures]
        assert mean_brightness[0] <= mean_brightness[1] <= mean_brightness[2]


class TestHDRProcessor:
    def test_full_pipeline(self):
        img = make_synthetic_image(64, 64)
        processor = HDRProcessor()
        results = processor.process_hdr_pipeline(img, exposure_ratios=[0.5, 1.0, 2.0])
        assert 'hdr_image' in results
        assert 'tone_mapped' in results
        assert results['hdr_image'].dtype == np.float32
        assert results['tone_mapped'].dtype == np.uint8

    def test_drago_method(self):
        img = make_synthetic_image(64, 64)
        processor = HDRProcessor()
        results = processor.process_hdr_pipeline(
            img, exposure_ratios=[0.5, 1.0, 2.0], tone_mapping_method='drago'
        )
        assert results['tone_mapped'].dtype == np.uint8
