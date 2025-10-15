# =================================================
# IMAGE WARPING and MOSAICING - PART B
# Main script 
# =================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.feature import corner_harris, peak_local_max
from main_3A import (computeH, get_panorama_bounding_box, warpImageBilinear)

# ========================================================
# B.1: Harris Corner Detection (Single Scale)

# provided 
def get_harris_corners(im, edge_discard=20, thresh=0.4, min_dist=10):
    """
    Args: 
    - im: grayscale image
    - edge_discard: number of pixels to discard near image edges
    - thresh: threshold for Harris response: cutoff_value = h.max() * thresh
    - min_dist: min number of pixels separating two detected corner points
    
    Returns: 
    - h: 2d array containing the h value of every pixel (same shape as original im)
    - coords of remaining corners (2 x n (ys, xs)) 
    """

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=min_dist, threshold_rel=thresh)

    # discard points on edge
    mask = (coords[:, 0] > edge_discard) & \
           (coords[:, 0] < im.shape[0] - edge_discard) & \
           (coords[:, 1] > edge_discard) & \
           (coords[:, 1] < im.shape[1] - edge_discard)
    corner_coords = coords[mask].T
    return h, corner_coords


def anms(corner_coords, h, num_points=500):
    """
    Args:
        corner_coords: (2, N) array of corner coordinates [y, x]
        h: Harris corner response map
        num_points: Number of corners to retain
    
    Returns:
        selected_coords: (2, num_points) array of selected corner coordinates
    """
    ys, xs = corner_coords
    strengths = h[ys, xs]
    
    # Compute suppression radius for each corner
    radii = np.full(len(ys), np.inf)
    
    for i in range(len(ys)):
        # Find all corners stronger than this one
        stronger_mask = strengths > strengths[i]
        if np.any(stronger_mask):
            # Compute distances to all stronger corners
            dy = ys[stronger_mask] - ys[i]
            dx = xs[stronger_mask] - xs[i]
            distances = np.sqrt(dy**2 + dx**2)
            # Minimum distance to a stronger corner
            radii[i] = np.min(distances)
    
    # Select top num_points corners with largest suppression radii
    selected_indices = np.argsort(radii)[-num_points:]
    selected_coords = np.array([ys[selected_indices], xs[selected_indices]])
    
    return selected_coords

def visualize_corners(im, anms_num_points=500):
    """
    Show Harris corners with and without ANMS.
    """

    print("\n Doing Harris corner detection")
    
    # Detect all Harris corners
    h, corners_all = get_harris_corners(im, edge_discard=20, thresh=0.1, min_dist=10)
    
    # Apply ANMS
    corners_anms = anms(corners_all, h, num_points=anms_num_points)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Without ANMS
    axes[0].imshow(im, cmap='gray')
    axes[0].plot(corners_all[1], corners_all[0], 'r+', markersize=8, markeredgewidth=1)
    axes[0].set_title(f'Harris Corners (All {corners_all.shape[1]} points)')
    axes[0].axis('off')
    
    # With ANMS
    axes[1].imshow(im, cmap='gray')
    axes[1].plot(corners_anms[1], corners_anms[0], 'r+', markersize=8, markeredgewidth=1)
    axes[1].set_title(f'After ANMS ({corners_anms.shape[1]} points)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    
    return corners_all, corners_anms

# ========================================================
# B.2: Feature Descriptor Extraction

def extract_feature_descriptors(im, corner_coords, patch_size=8, window_size=40, sample_spacing=5):
    """    
    Args:
    - corners: 2 x N array of corner coordinates (ys, xs)
    - patch_size: size of final descriptor patch 
    - window_size: size of window to sample around each corner  
    - sample_spacing: interval to sample pixels at for final 8x8
    
    Returns:
    - descriptors: M x (patch_size*patch_size) array of descriptors
    - valid_corner_coords: 2 x M array of coords that could be sampled
    """
    
    ys, xs = corner_coords
    descriptors = []
    valid_corner_coords = [] # coords

    half_window = window_size // 2

    for y, x in zip(ys, xs):

        # Check if the 40x40 window fits inside image
        if y - half_window < 0 or y + half_window >= im.shape[0] or \
           x - half_window < 0 or x + half_window >= im.shape[1]:
            continue
        
        # Extract 40x40 window around the corner
        window = im[y-half_window:y+half_window, x-half_window:x+half_window]

        # Downsample window to 8x8 patch using sample_spacing (pick every s-th pixel)
        start = half_window - (patch_size//2)*sample_spacing
        patch = window[start:start+patch_size*sample_spacing:sample_spacing,
                       start:start+patch_size*sample_spacing:sample_spacing]
        
        # Flatten and normalize (bias/gain normalization)
        vec = patch.flatten().astype(np.float32)
        vec -= np.mean(vec)
        std = np.std(vec)
        if std > 1e-5:
            vec /= std
        else:
            continue # skip patches with almost zero variance
        
        descriptors.append(vec)
        valid_corner_coords.append([y, x])
    
    descriptors = np.array(descriptors)                     # M x (patch_size*patch_size)
    valid_corner_coords = np.array(valid_corner_coords).T   # 2 x M 

    print(f"Extracted {len(descriptors)} feature descriptors (8x8)")
    print(f"Descriptor shape: {descriptors.shape}")

    return descriptors, valid_corner_coords


def visualize_feature_descriptors(descriptors, valid_corners): 
    
    print("\n Getting feature descriptors")
    
    # Visualize several descriptors
    num_to_show = 8
    fig, axes = plt.subplots(4, 2, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < num_to_show and i < len(descriptors):
            # Reshape descriptor to 8x8 and normalize for display
            desc_img = descriptors[i].reshape(8, 8)
            ax.imshow(desc_img, cmap='gray')
            y, x = valid_corners[:, i]
            ax.set_title(f'Corner ({int(x)}, {int(y)})', fontsize=9)
        ax.axis('off')
    
    plt.suptitle('Sample Feature Descriptors (8x8 normalized patches)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return descriptors, valid_corners

# ========================================================
# B.3: Feature Matching

import numpy as np
from scipy.spatial.distance import cdist

def match_features(desc1, desc2, ratio_thresh=0.8):
    """    
    Args:
    - desc1: N1 x D array of descriptors from image 1
    - desc2: N2 x D array of descriptors from image 2
    - ratio_thresh: threshold for Lowe's ratio test (lower = best match must be much better than second best match)
    
    Returns:
    - matches: list of (i, j) tuples indicating matched descriptor indices
    - distances: list of distances for each match
    """
    print("Getting matches")

    # Compute pairwise Euclidean distances between descriptors
    dists = cdist(desc1, desc2)

    matches = []
    distances = []
    for i in range(dists.shape[0]):
        sorted_idx = np.argsort(dists[i])
        # Lowe's ratio test
        best, second_best = sorted_idx[0], sorted_idx[1]
        ratio = dists[i, best] / (dists[i, second_best] + 1e-10)
        if ratio < ratio_thresh:
            matches.append((i, best))
            distances.append(dists[i, best])
    
    print(f"Found {len(matches)} matches using Lowe's ratio test (threshold={ratio_thresh})")

    return matches, distances

def visualise_matches(im1, im2, coords1, coords2, matches, max_display=20):
    
    print("Visualising matches")
    
    # Visualize matches
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3))
    canvas[:h1, :w1] = im1 if im1.ndim == 3 else np.stack([im1]*3, axis=2)
    canvas[:h2, w1:] = im2 if im2.ndim == 3 else np.stack([im2]*3, axis=2)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(canvas)
    
    # Visualise subset of matches
    display_matches = matches[:max_display] if len(matches) > max_display else matches
    
    for i, j in display_matches:
        y1, x1 = coords1[:, i]
        y2, x2 = coords2[:, j]
        
        plt.plot([x1, x2 + w1], [y1, y2], 'g-', linewidth=0.8, alpha=0.6)
        plt.plot(x1, y1, 'ro', markersize=3)
        plt.plot(x2 + w1, y2, 'ro', markersize=3)
    
    plt.title(f'Feature Matches (showing {len(display_matches)} of {len(matches)} total)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return matches

# ========================================================
# B.4: RANSAC for Robust Homography

def compute_homography_from_matches(coords1, coords2, matches, n_iters=2000, inlier_threshold=3.0):
    """    
    Args:
        coords1: (2, N1) array of corner coordinates from image 1 [y, x]
        coords2: (2, N2) array of corner coordinates from image 2 [y, x]
        matches: List of (i, j) tuples indicating matched descriptor indices
        n_iters: Number of RANSAC iterations
        inlier_threshold: Threshold for pixel diff between projected points and actual correspondences (to be considered inlier)
    
    Returns:
        best_H: Best homography matrix
        best_inliers: Indices of inlier matches
    """
    
    print("Computing homographies via RANSAC")

    # Get matched point coordinates
    pts1 = coords1[:, [m[0] for m in matches]].T  # (N, 2) in (y, x)
    pts2 = coords2[:, [m[1] for m in matches]].T
    
    # Convert to (x, y) for homography computation
    pts1 = np.fliplr(pts1)
    pts2 = np.fliplr(pts2)

    # RANSAC 
    best_inliers = []
    best_H = None
    n_pts = len(pts1)
    for iter in range(n_iters):
        # Randomly sample 4 points
        idx = np.random.choice(n_pts, 4, replace=False)
        
        # Compute homography from these 4 points
        H = computeH(pts1[idx], pts2[idx])

        # Apply homography to project im1 points to im2 - code from part A 
        pts1_h = np.hstack([pts1, np.ones((n_pts, 1))])  # append 1 to each point [(x, y, 1),...]
        projected = (H @ pts1_h.T).T
        projected = projected[:, :2] / (projected[:, 2, np.newaxis] + 1e-10)

        # Compute projection errors
        errors = np.linalg.norm(projected - pts2, axis=1)
        
        # Count inliers
        inliers = np.where(errors < inlier_threshold)[0]

        # Update best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    # Refine homography using all inliers
    best_H = computeH(pts1[best_inliers], pts2[best_inliers])
    print(f"RANSAC completed: {len(inliers)} inliers out of {len(matches)} matches")
    
    return best_H, best_inliers

def visualize_ransac_results(im1, im2, coords1, coords2, matches, inliers):
    """    
    Args:
        coords1: (2, N1) corner coordinates from image 1 [y, x]
        coords2: (2, N2) corner coordinates from image 2 [y, x]
        matches: List of (i, j) match tuples
        inliers: Array of inlier indices
    """
    print("Visualising RANSAC inliers")

    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
    # Create side-by-side image
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3))
    canvas[:h1, :w1] = im1 if im1.ndim == 3 else np.stack([im1]*3, axis=2)
    canvas[:h2, w1:] = im2 if im2.ndim == 3 else np.stack([im2]*3, axis=2)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(canvas)
    
    # Convert inliers to set for fast lookup
    inlier_set = set(inliers)
    
    # Draw all matches
    for idx, (i, j) in enumerate(matches):
        y1, x1 = coords1[:, i]
        y2, x2 = coords2[:, j]
        
        if idx in inlier_set:
            # Inliers in green
            plt.plot([x1, x2 + w1], [y1, y2], 'g-', linewidth=0.7, alpha=0.5)
            plt.plot(x1, y1, 'go', markersize=4)
            plt.plot(x2 + w1, y2, 'go', markersize=4)
        else:
            # Outliers in red (fainter)
            plt.plot([x1, x2 + w1], [y1, y2], 'r-', linewidth=0.7, alpha=0.5)
    
    plt.title(f'RANSAC Results: {len(inliers)} inliers (green) / {len(matches)} total matches', 
              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_panorama(images, homographies):
    """    
    Args:
        images: List of images (each as np.ndarray with shape (h, w, 3) or (h, w))
        homographies: List of 3x3 homography matrices (one per image, mapping to reference)
    
    Returns:
        panorama: Blended panorama image
    """
    print(f"\nCreating panorama from {len(images)} images")
        
    # Get bounding box
    H, W, T = get_panorama_bounding_box(images, homographies)
    
    # Num and denom for weighted averaging
    num = np.zeros((H, W, 3), dtype=np.float32)
    den = np.zeros((H, W, 3), dtype=np.float32)
    
    # Warp each image into canvas
    for i, (im, H_i) in enumerate(zip(images, homographies)):
        print(f"Warping image {i+1}/{len(images)}...")
        H_final = T @ H_i
        im_warped, mask = warpImageBilinear(im, H_final, out_shape=(H, W))
        
        # Expand mask to 3 channels
        m3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
        
        # Accumulate
        num += im_warped * m3
        den += m3
        
        del im_warped, mask, m3  # free memory

    # Blend
    panorama = num / np.maximum(den, 1e-8)
    panorama = np.clip(panorama, 0, 1)
    print("Panorama blending done")
    
    return panorama

def visualize_panorama(panorama, title='Panorama (w Auto Feature Matching)'):
 
    plt.figure(figsize=(20, 10))
    plt.imshow(panorama, cmap='gray' if panorama.ndim == 2 else None)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ========================================================
# MAIN

im1 = img_as_float(io.imread("inputs/wheeler1.JPG"))
im2 = img_as_float(io.imread("inputs/wheeler2.JPG"))
im3 = img_as_float(io.imread("inputs/wheeler3.JPG"))

gray1 = color.rgb2gray(im1) if im1.ndim == 3 else im1
gray2 = color.rgb2gray(im2) if im2.ndim == 3 else im2
gray3 = color.rgb2gray(im3) if im3.ndim == 3 else im3

# B.1: Harris Corner Detection
corners1_all, corners1_anms = visualize_corners(gray1, anms_num_points=200)
corners2_all, corners2_anms = visualize_corners(gray2, anms_num_points=200)
corners3_all, corners3_anms = visualize_corners(gray3, anms_num_points=200)

# B.2: Feature Descriptor Extraction
desc1, coords1 = extract_feature_descriptors(gray1, corners1_anms)
desc2, coords2 = extract_feature_descriptors(gray2, corners2_anms)
desc3, coords3 = extract_feature_descriptors(gray3, corners3_anms)
visualize_feature_descriptors(desc1, coords1)

# B.3: Feature Matching
matches_12, _ = match_features(desc1, desc2)
visualise_matches(gray1, gray2, coords1, coords2, matches_12)
matches_32, _ = match_features(desc3, desc2)
visualise_matches(gray3, gray2, coords3, coords2, matches_32)

# B.4: RANSAC and Panorama
# compute homography via RANSAC
H12, inliers_12 = compute_homography_from_matches(coords1, coords2, matches_12, n_iters=2000, inlier_threshold=4.0)
visualize_ransac_results(im1, im2, coords1, coords2, matches_12, inliers_12)
H32, inliers_32 = compute_homography_from_matches(coords3, coords2, matches_32, n_iters=2000, inlier_threshold=4.0)
visualize_ransac_results(im3, im2, coords3, coords2, matches_32, inliers_32)

# create panorama
panorama = create_panorama([im1, im2, im3], [H12, np.eye(3), H32])
visualize_panorama(panorama)
