"""
Image Alignment Module
======================

Implements feature-based image alignment for HDR processing.
Based on Chapter 13 of "Mobile Computational Photography".

The alignment process:
1. Detect features using ORB (Oriented FAST and Rotated BRIEF)
2. Match features between reference and target images
3. Estimate homography transformation
4. Warp target image to align with reference

This addresses the camera shake and object motion challenges in multi-exposure HDR.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class ImageAligner:
    """
    Feature-based image alignment for HDR processing.
    
    Uses ORB feature detection and FLANN-based matching to align
    multiple exposure images before HDR fusion.
    """
    
    def __init__(self, n_features: int = 1000, match_ratio: float = 0.7):
        """
        Initialize the image aligner.
        
        Args:
            n_features: Number of ORB features to detect (default: 1000)
            match_ratio: Ratio threshold for good matches (default: 0.7)
                        Lower values are more restrictive
        """
        self.n_features = n_features
        self.match_ratio = match_ratio
        self.orb = cv2.ORB_create(nfeatures=n_features)
        
        # FLANN matcher parameters for feature matching
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def align_images(self, images: List[np.ndarray], reference_idx: int = 2) -> List[np.ndarray]:
        """
        Align multiple exposure images to a reference image.
        
        This is a critical step in HDR processing as even slight camera
        movement between exposures can cause ghosting artifacts in the
        final HDR image.
        
        Args:
            images: List of exposure images to align
            reference_idx: Index of the reference image (others align to this)
            
        Returns:
            List of aligned images (reference image unchanged, others warped)
        """
        if len(images) < 2:
            return images
        
        print("🎯 Performing image alignment...")
        reference = images[reference_idx]
        aligned_images = [reference.copy()]
        
        # Detect features in reference image
        kp_ref, des_ref = self.orb.detectAndCompute(reference, None)
        
        if des_ref is None:
            print("   Warning: No features detected in reference image")
            return images
        
        # Align each image to the reference
        for i, img in enumerate(images):
            if i == reference_idx:
                continue  # Skip reference image itself
            
            aligned_img = self._align_to_reference(img, kp_ref, des_ref, reference.shape)
            if aligned_img is not None:
                aligned_images.append(aligned_img)
            else:
                print(f"   Warning: Alignment failed for image {i}, using original")
                aligned_images.append(img)
        
        print(f"   Completed alignment for {len(aligned_images)} images")
        return aligned_images
    
    def _align_to_reference(self, target_img: np.ndarray, 
                           kp_ref: List, des_ref: np.ndarray,
                           ref_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Align a single target image to the reference.
        
        Args:
            target_img: Image to align
            kp_ref: Reference keypoints
            des_ref: Reference descriptors
            ref_shape: Shape of reference image (height, width)
            
        Returns:
            Aligned image or None if alignment fails
        """
        # Detect features in target image
        kp_target, des_target = self.orb.detectAndCompute(target_img, None)
        
        if des_target is None:
            return None
        
        # Match features
        try:
            matches = self.flann.knnMatch(des_ref, des_target, k=2)
        except Exception:
            return None
        
        # Apply ratio test (Lowe's ratio test) to filter good matches
        # This helps reject ambiguous matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Good match if distance is significantly smaller than second best
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        # Need at least 10 good matches to estimate homography
        if len(good_matches) < 10:
            return None
        
        # Extract matched point coordinates
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography using RANSAC (robust to outliers)
        # Homography models perspective transformation between images
        try:
            homography, mask = cv2.findHomography(
                dst_pts, src_pts,
                cv2.RANSAC,
                ransacReprojThreshold=5.0  # Maximum allowed reprojection error
            )
            
            if homography is not None:
                # Apply perspective transformation
                h, w = ref_shape[:2]
                aligned = cv2.warpPerspective(target_img, homography, (w, h))
                print(f"   Successfully aligned image with {len(good_matches)} matches")
                return aligned
        except Exception:
            pass
        
        return None

