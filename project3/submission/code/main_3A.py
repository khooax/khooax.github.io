
# =================================================
# IMAGE WARPING and MOSAICING
# Main script 
# =================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color 
import cv2
import os
import skimage.transform as tf

# ========================================================
# A.2: Recover Homographies

def computeH(im1_pts, im2_pts):
    """
    Args: 
        im1_pts: nx2 array of points in im1
        im2_pts: nx2 array of corresponding points in im2
    
    Returns: 
        H: 3x3 matrix
    """
    n = im1_pts.shape[0]
    A = []
    b = []
    for i in range(n):
        x, y = im1_pts[i]
        x_p, y_p = im2_pts[i]

        # first row
        A.append([x, y, 1, 0, 0, 0, -x_p * x, -x_p * y])
        b.append(x_p)

        # second row
        A.append([0, 0, 0, x, y, 1, -y_p * x, -y_p * y])
        b.append(y_p)

    A = np.array(A)
    b = np.array(b)

    # Solve Ah = b using least squares
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H = np.append(h, 1).reshape((3, 3))
    
    return H


def get_points(im1, im2=None, n=12): 

    plt.imshow(im1)
    plt.title(f"Click {n} points in Image 1")
    im1_pts = np.array(plt.ginput(n, timeout=0))  # shape (n,2)
    plt.close()

    if im2 is not None: 
        plt.imshow(im2)
        plt.title(f"Click {n} corresponding points in Image 2")
        im2_pts = np.array(plt.ginput(n, timeout=0))
        plt.close()
    else: 
        im2_pts = None

    return im1_pts, im2_pts

def apply_homography(H, pts):
    """
    Apply homography H to points pts. 
    Take a set of 2D points in image 1 and compute where they land in image 2 using H.
    
    Args:
        H: 3x3 matrix
        pts: (N, 2) array of points [x, y]
    
    Returns:
        transformed_pts: (N, 2) array of transformed points
    """
    n = pts.shape[0]
    
    # build homog_pts by appending a 1 to each point [(x, y, 1),...]
    homog_pts = np.hstack([pts, np.ones((n,1))])   # (n,3)
    # matmul with H 
    projected = (H @ homog_pts.T).T                # (n,3)
    # proj_homog_i is [u, v, w]. get actual 2D pt [u/w, v/w].
    projected = projected / projected[:, [2]]       
    return projected[:, :2]

def plot_correspondences(img1, img2, pts1, pts2, title=None, n_lines=20):
    """
    pts1, pts2: Nx2 arrays of (x, y) coords
    """

    # Convert to uint8 if float images (preserving color)
    if img1.dtype != np.uint8:
        img1 = np.clip(img1 * 255, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2 * 255, 0, 255).astype(np.uint8)

    # Combine horizontally
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # Shift right image points by image width
    pts2_shifted = pts2.copy()
    pts2_shifted[:, 0] += w1

    # Random subset of correspondences
    n = min(n_lines, len(pts1))
    idx = np.random.choice(len(pts1), n, replace=False)

    # Draw matches
    for i in idx:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt1 = tuple(np.round(pts1[i]).astype(int))
        pt2 = tuple(np.round(pts2_shifted[i]).astype(int))
        cv2.line(canvas, pt1, pt2, color, 3)         # thicker line
        cv2.circle(canvas, pt1, 10, color, -1)       # larger dot
        cv2.circle(canvas, pt2, 10, color, -1)

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas) 
    if title:
        plt.title(title, fontsize=16)
    plt.axis("off")
    plt.show()

#====Main ====
if __name__ == "__main__":
    im1 = img_as_float(io.imread("inputs/interior1.JPG")[..., :3])  
    im2 = img_as_float(io.imread("inputs/interior2.JPG")[..., :3])  

    # Get correspondences
    im1_pts, im2_pts = get_points(im1, im2, n=12)
    plot_correspondences(im1, im2, im1_pts, im2_pts, title=f"Correspondences")

    # Compute homography
    H = computeH(im1_pts, im2_pts)
    print("Recovered homography:\n", H)

# ========================================================
# A.3: Warp the Images

def get_destination_bounding_box(H, src_w, src_h):
    """
    Compute destination bounding box after applying homography.
    
    Args:
        H: 3x3 homography matrix
        src_w: source image width
        src_h: source image height
    
    Returns:
        min_x, min_y, max_x, max_y: bounding box coordinates
    """
    corners = np.array([[0,0], [src_w-1,0], [src_w-1,src_h-1], [0,src_h-1]], dtype=float)
    dest = apply_homography(H, corners) # (4, 2) projected bounding box
    min_x = np.floor(dest[:,0].min()).astype(int)
    min_y = np.floor(dest[:,1].min()).astype(int)
    max_x = np.ceil(dest[:,0].max()).astype(int)
    max_y = np.ceil(dest[:,1].max()).astype(int)
    return min_x, min_y, max_x, max_y

def warpImageNearestNeighbor(im, H, fill_value=0, out_shape=None):
    """
    Args: 
        im: (H, W) or (H, W, C)
        
    Returns: 
        out: warped image , same dtype as im
        alpha: binary mask (1 where out has a valid sampled pixel, 0 otherwise)
    """
    im = np.asarray(im)
    src_h, src_w = im.shape[:2]
    channels = im.shape[2]
    
    if out_shape is None:
        # Predict dest bounding box and output shape
        min_x, min_y, max_x, max_y = get_destination_bounding_box(H, src_w, src_h)
        out_w = max_x - min_x + 1
        out_h = max_y - min_y + 1
    else: 
        out_h, out_w = out_shape
        xs = np.arange(out_w)
        ys = np.arange(out_h)
        min_x, min_y = 0, 0

    # Initialise output arrays
    out = np.full((out_h, out_w, channels), fill_value, dtype=im.dtype)
    alpha = np.zeros((out_h, out_w), dtype=np.uint8)

    # Destination grid  
    xs = np.arange(min_x, max_x+1) # destination x-coords 
    ys = np.arange(min_y, max_y+1) # destination y-coords 
    X, Y = np.meshgrid(xs, ys)  # (out_h, out_w), X = x-coords repeated across rows, Y = y-coords repeated across columns, (X[i,j], Y[i,j]) = a point
    # flatten 
    Xf = X.ravel()
    Yf = Y.ravel()
    dest_pts = np.vstack([Xf, Yf]).T  # (N,2), dest_pts = [[Xf[i], Yf[i]] ...]

    # Inverse mapping (dest coords to src) 
    H_inv = np.linalg.inv(H) 
    # for each dest point [x_d,y_d,1], compute H_inv @ [xd,yd,1]^T and normalize by the third coordinate
    src_pts = apply_homography(H_inv, dest_pts) # (n, 2)
    src_x = src_pts[:,0]
    src_y = src_pts[:,1]

    # Nearest neighbor sampling
    # round src coords to nearest integer pixel
    src_xr = np.rint(src_x).astype(int)
    src_yr = np.rint(src_y).astype(int)
    # check if mapped coord actually falls within original src bounds 
    valid = (src_xr >= 0) & (src_xr < src_w) & (src_yr >= 0) & (src_yr < src_h) # 1D boolean array
    # translate dest x&y-coords so that min_x and min_y is treated as 0
    out_cols = (Xf - min_x).astype(int) 
    out_rows = (Yf - min_y).astype(int)
    # assign pixel values from src image to output img (out), based on the valid mapping
    out[out_rows[valid], out_cols[valid], :] = im[src_yr[valid], src_xr[valid], :]
    alpha[out_rows[valid], out_cols[valid]] = 1
    return out, alpha

def warpImageBilinear(im, H, fill_value=0, out_shape=None):
    """
    Args:
        im: input image (h, w, 3) or (h, w)
        H: 3x3 homography matrix
        fill_value: Value for pixels outside source image
        out_shape: Optional (height, width) for output
    
    Returns:
        warped_image: Warped image
        alpha_mask: Binary mask indicating valid pixels
    """

    im = np.asarray(im)
    src_h, src_w = im.shape[:2]

    # Handle grayscale images
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
        was_grayscale = True
    else:
        was_grayscale = False
    
    channels = im.shape[2]
    
    if out_shape is None:
        # Predict dest bounding box and output shape
        min_x, min_y, max_x, max_y = get_destination_bounding_box(H, src_w, src_h)
        out_w = max_x - min_x + 1
        out_h = max_y - min_y + 1
    else: 
        out_h, out_w = out_shape
        min_x, min_y = 0, 0
        max_x, max_y = out_w - 1, out_h - 1

    # Initialise output array
    out = np.full((out_h, out_w, channels), fill_value, dtype=im.dtype)
    alpha = np.zeros((out_h, out_w), dtype=np.uint8)

    # Destination grid  
    xs = np.arange(min_x, max_x+1) # destination x-coords 
    ys = np.arange(min_y, max_y+1) # destination y-coords 
    X, Y = np.meshgrid(xs, ys)  # (out_h, out_w), X = x-coords repeated across rows, Y = y-coords repeated across columns, (X[i,j], Y[i,j]) = a point
    Xf = X.ravel() # flatten 
    Yf = Y.ravel()
    dest_pts = np.vstack([Xf, Yf]).T  # (N,2), dest_pts = [[Xf[i], Yf[i]] ...]

    # Inverse mapping (dest coords to src) 
    H_inv = np.linalg.inv(H) 
    # for each dest point [x_d,y_d,1], compute H_inv @ [xd,yd,1]^T and normalize by the third coordinate
    src_pts = apply_homography(H_inv, dest_pts) # (n, 2)
    sx = src_pts[:,0]
    sy = src_pts[:,1]

    # Bilinear sampling
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = sx - x0  # fractional part in x
    wy = sy - y0  # fractional part in y

    # Valid when all four neighbors are inside bounds
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < src_w) & (y1 < src_h)
    out_cols = (Xf - min_x).astype(int)
    out_rows = (Yf - min_y).astype(int)

    # Work per-channel with vectorized indexing
    for c in range(channels):
        I00 = np.zeros_like(sx); I10 = np.zeros_like(sx)
        I01 = np.zeros_like(sx); I11 = np.zeros_like(sx)
        channel = im[:,:,c]
        I00[valid] = channel[y0[valid], x0[valid]]
        I10[valid] = channel[y0[valid], x1[valid]]
        I01[valid] = channel[y1[valid], x0[valid]]
        I11[valid] = channel[y1[valid], x1[valid]]
        val = (1-wx)*(1-wy)*I00 + wx*(1-wy)*I10 + (1-wx)*wy*I01 + wx*wy*I11
        out[out_rows[valid], out_cols[valid], c] = val[valid]

    alpha[out_rows[valid], out_cols[valid]] = 1
    # Convert back to original dtype
    out = np.clip(out, 0, 255).astype(im.dtype)

    # Handle grayscale output
    if was_grayscale:
        out = out[:, :, 0]
        
    return out, alpha

# === Main === 
if __name__ == "__main__":

    im = img_as_float(io.imread("inputs/caljournal.JPG")[..., :3])  

    # Click the corners
    im1_pts, _ = get_points(im, n=4)

    # Define target rectangle 
    W, H = 148*4, 210*4   # A5 journal 
    # W, H = 200, 200   # square checkerboard 
    im2_pts = np.array([
        [0, 0],         # top-left
        [W-1, 0],       # top-right
        [W-1, H-1],     # bottom-right
        [0, H-1]        # bottom-left
    ])

    # Compute H 
    H = computeH(im1_pts, im2_pts)

    # Apply warping
    rect_nn, mask_nn = warpImageNearestNeighbor(im, H)
    rect_bl, mask_bl = warpImageBilinear(im, H)

    # Visualise
    fig, axes = plt.subplots(1, 3)
    axes = axes.ravel()
    axes[0].imshow(im, cmap='gray'); axes[0].set_title("Original") 
    axes[1].imshow(rect_nn, cmap='gray'); axes[1].set_title("Rectified (NN)") 
    axes[2].imshow(rect_bl, cmap='gray'); axes[2].set_title("Rectified (BL)") 
    for i in range(3): 
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# ========================================================
# A.4: Blend images into Mosaic

def get_homographies_to_ref(computeH, correspondences):
    """
    Compute homographies mapping all images into reference image coords
    computeH: function that takes (imA_pts, imB_pts) and returns HAtoB
    correspondences: dict containing matched points between image pairs.
        {('1','2'): ..., ('2','3'): ...}
    Returns list of homographies [H1to2, H2to2, H3to2...]
    """

    # Forward direction: 1->2->3->4
    H12 = computeH(*correspondences['1_2'])
    H23 = computeH(*correspondences['2_3'])

    # Compose into reference frame (image 4)
    H1to2 = H12
    H2to2 = np.eye(3)
    H3to2 = np.linalg.inv(H23)
    homographies = [H1to2, H2to2, H3to2]

    # Normalize so bottom-right entry = 1
    for i in range(len(homographies)):
        homographies[i] /= homographies[i][2, 2]

    return homographies

def get_panorama_bounding_box(images, homographies):
    """    
    Args:
        images: list of np.ndarray images (each shape (h, w, 3) or (h, w))
        homographies: list of 3x3 homographies mapping each image to reference
    
    Returns:
        H: int, panorama height
        W: int, panorama width
        T: 3x3 translation homography that shifts everything so top-left corner is (0, 0)
    """
    all_corners = []

    for im, H in zip(images, homographies):
        h, w = im.shape[:2]

        # Corners of the source image (homog coordinates)
        corners = np.array([
            [0,   0,   1],
            [w-1, 0,   1],
            [w-1, h-1, 1],
            [0,   h-1, 1]
        ]).T  # shape (3,4)

        # Warp corners
        warped = H @ corners
        warped /= warped[2,:]  # normalize
        all_corners.append(warped[:2,:])  # (2,4)

    all_corners = np.hstack(all_corners)  # (2, 4*num_images)
    xs, ys = all_corners[0,:], all_corners[1,:]

    # Bounding box
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    W = int(np.ceil(x_max - x_min))
    H = int(np.ceil(y_max - y_min))

    # Get translation matrix to shift everything into positive coords
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    return H, W, T

# === Main ====
if __name__ == "__main__":

    im1 = img_as_float(io.imread("inputs/interior1.JPG")[..., :3])  
    im2 = img_as_float(io.imread("inputs/interior2.JPG")[..., :3])  
    im3 = img_as_float(io.imread("inputs/interior3.JPG")[..., :3])  
    images = [im1, im2, im3]
    scale_factor = 0.4  # downscale
    images = [tf.rescale(im, scale_factor, channel_axis=2, anti_aliasing=True) for im in images]

    correspondences = {}

    corr_dir = "correspondences"  # relative to submission
    os.makedirs(corr_dir, exist_ok=True) 

    for i in range(len(images) - 1):
        key = f"{i+1}_{i+2}"
        fname = os.path.join(corr_dir, f"corresp_interior_{key}.npz")

        if os.path.exists(fname):
            data = np.load(fname)
            pts_src, pts_dst = data["pts_src"], data["pts_dst"]
            correspondences[key] = (pts_src, pts_dst)
            print(f"Loaded saved correspondences for pair {key}")
        else:
            print(f"Click {key}: select points in image {i+1} then corresponding points in image {i+2}")
            pts_src, pts_dst = get_points(images[i], images[i+1], n=16)
            correspondences[key] = (pts_src, pts_dst)
            np.savez(fname, pts_src=pts_src, pts_dst=pts_dst)
            print(f"Saved correspondences to {fname}")

        plot_correspondences(images[i], images[i+1], pts_src, pts_dst, title=f"Correspondences {key}")

    # Compute homographies to ref image
    homographies = get_homographies_to_ref(computeH, correspondences) # returns [H1to2, H2to2, H3to2]

    # Get bounding box 
    H, W, T = get_panorama_bounding_box(images, homographies)
    num = np.zeros((H, W, 3), dtype=np.float32) # numerator for weighted avg: sum of all pixel colors where images overlap
    den = np.zeros((H, W, 3), dtype=np.float32) # denom for weighted avg: how many images overlap there

    # Warp each image into (H, W) canvas
    mosaic_stack = []
    alpha_stack = []

    for i, (im, H_i) in enumerate(zip(images, homographies)):
        H_final = T @ H_i
        im_warped, mask = warpImageBilinear(im, H_final, out_shape=(H, W))
        m3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
        num += im_warped * m3
        den += m3
        del im_warped, mask, m3  # free memory
        print(f"Blended image {i+1}/{len(images)}")

    mosaic_final = num / np.maximum(den, 1e-8)
    mosaic_final = np.clip(mosaic_final, 0, 1)
    print("Completed panorama") # DEBUG 

    # visualise 
    plt.figure(figsize=(12,8))
    plt.imshow(mosaic_final)
    plt.axis("off")
    plt.show()

    plt.imsave("mosaic.png", mosaic_final)