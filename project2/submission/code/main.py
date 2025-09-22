# =================================================
# Fun with Filters and Frequencies 
# Main script 
# =================================================

import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib 
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import time
from skimage import io, img_as_float, color 
from skimage.color import rgb2gray
import cv2
from project2.submission.code.align_image_code import align_images

# ========================================================
# Part 1.1: Convolutions from Scratch

def conv2d_four_loops(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    k_flipped = np.flipud(np.fliplr(kernel))
    pad_h = kH // 2
    pad_w = kW // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 
                    mode='constant', constant_values=0.0)
    out_image = np.zeros((H, W), dtype=np.float64) # (H, W)

    for i in range(H):
        for j in range(W):
            pix_output = 0.0
            for m in range(kH):
                for n in range(kW):
                    pix_output += padded[i + m, j + n] * k_flipped[m, n]
            out_image[i, j] = pix_output
    return out_image

def conv2d_two_loops(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    k_flipped = np.flipud(np.fliplr(kernel))
    pad_h = kH // 2
    pad_w = kW // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 
                    mode='constant', constant_values=0.0)
    out_image = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            patch = padded[i:i + kH, j:j + kW] # (kH, kW) 
            out_image[i, j] = np.sum(patch * k_flipped) 
    return out_image

def compare_with_scipy(image, kernel):
    t0 = time.time()
    out4 = conv2d_four_loops(image, kernel)
    t1 = time.time()
    out2 = conv2d_two_loops(image, kernel)
    t2 = time.time()
    out_scipy = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    t3 = time.time()

    # compare timings
    print(f"timings: 4-loops {t1-t0:.4f}s, 2-loops {t2-t1:.4f}s, scipy {t3-t2:.4f}s")
    # compare pixel differences
    print(f"max pixel differences: 4-loops vs scipy = {np.max(np.abs(out4-out_scipy)):.3e}")
    print(f"max pixel differences: 2-loops vs scipy = {np.max(np.abs(out2-out_scipy)):.3e}")
    return out4, out2, out_scipy

# ----- Main  -----
print("Starting 1.1")
img = img_as_float(io.imread("submission/inputs/1.1_selfie.jpg")[..., :3]) # (H,W,3)
img = rgb2gray(img)

box9 = np.ones((9, 9), dtype=np.float64) / 81.0
blur = conv2d_two_loops(img, box9)

print("\n===== Convolution implementation comparisons (9x9 box filter) =====")
out4, out2, out_scipy = compare_with_scipy(img, box9)

# visualize
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(blur, cmap='gray'); plt.title("Blurred 9x9"); plt.axis('off')
plt.show()

# ========================================================
# Part 1.2: Finite Difference Operator

def binarize_gradient(grad_mag, fraction=0.2):
    threshold = fraction * np.max(grad_mag)
    binary_edge = grad_mag >= threshold
    return binary_edge

# ----- Main ----- 
print("Starting 1.2")

img = img_as_float(io.imread("submission/inputs/1.2_cameraman.png")[..., :3]) # (H,W,3)
img = rgb2gray(img)

Dx = np.array([[1, 0, -1]], dtype=np.float64)
Dy = np.array([[1], [0], [-1]], dtype=np.float64)

# convolution and calculate grad_mag
gx = convolve2d(img, Dx, mode='same', boundary='fill', fillvalue=0) # (H, W)
gy = convolve2d(img, Dy, mode='same', boundary='fill', fillvalue=0) # (H, W)
grad_mag = np.sqrt(gx**2 + gy**2)

# binarize grad mag to get binary edges 
edges_10 = binarize_gradient(grad_mag, 0.10)
edges_20 = binarize_gradient(grad_mag, 0.20)

# visualize
fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes = axes.ravel()
axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original") 
axes[1].imshow(gx, cmap='gray'); axes[1].set_title("Gx (horizontal)") 
axes[2].imshow(gy, cmap='gray'); axes[2].set_title("Gy (vertical)") 
axes[3].imshow(grad_mag, cmap='gray'); axes[3].set_title("Gradient magnitude image") 
axes[4].imshow(edges_10, cmap='gray'); axes[4].set_title("Binarized edge image (thresh=0.10)") 
axes[5].imshow(edges_20, cmap='gray'); axes[5].set_title("Binarized edge image (thresh=0.20)")
for i in range(6): 
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# ========================================================
# Part 1.3: Derivative of Gaussian (DoG) Filter

def gaussian_2d(kernel_size=5, sigma=1.0):
    g1d = cv2.getGaussianKernel(kernel_size, sigma)
    g2d = g1d @ g1d.T
    return g2d

# ----- Main -----
print("Starting 1.3")

img = img_as_float(io.imread("submission/inputs/1.2_cameraman.png")[..., :3]) # (H,W,3)
img = rgb2gray(img)

Dx = np.array([[1, 0, -1]], dtype=np.float64)
Dy = np.array([[1], [0], [-1]], dtype=np.float64)

# edges without smoothing
gx = convolve2d(img, Dx, mode='same', boundary='fill', fillvalue=0) # (H, W)
gy = convolve2d(img, Dy, mode='same', boundary='fill', fillvalue=0) # (H, W)
grad_mag = np.sqrt(gx**2 + gy**2)
edges_raw = binarize_gradient(grad_mag, 0.10)

# edges after gaussian smoothing (seperate steps) 
G = gaussian_2d(9, 2.0)
smoothed = convolve2d(img, G, boundary='fill', fillvalue=0)
gx = convolve2d(smoothed, Dx, mode='same', boundary='fill', fillvalue=0) # (H, W)
gy = convolve2d(smoothed, Dy, mode='same', boundary='fill', fillvalue=0) # (H, W)
grad_mag = np.sqrt(gx**2 + gy**2)
edges_smooth = binarize_gradient(grad_mag, 0.10)

# DoG filters (combined into 1 step)
DoG_x = convolve2d(G, Dx, mode='same')   
DoG_y = convolve2d(G, Dy, mode='same')  
gx_DoG = convolve2d(img, DoG_x, mode='same', boundary='fill', fillvalue=0)
gy_DoG = convolve2d(img, DoG_y, mode='same', boundary='fill', fillvalue=0)
grad_mag = np.sqrt(gx_DoG**2 + gy_DoG**2)
edges_DoG = binarize_gradient(grad_mag, 0.10)

# visualize
fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes = axes.ravel()
axes[0].imshow(edges_raw, cmap='gray'); axes[0].set_title("Edges (without Gaussian)")
axes[1].imshow(edges_smooth, cmap='gray'); axes[1].set_title("Edges (Gaussian then Dx Dy)") 
axes[2].imshow(edges_DoG, cmap='gray'); axes[2].set_title("Edges (DoG)") 
for i in range(3):
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# ========================================================
# Part 2.1: Image "Sharpening"

def unsharp_mask_filter(img, kernel_size=9, gaussian_std=2.0, alpha=1.0):

    # gaussian filter
    G = gaussian_2d(kernel_size, gaussian_std)

    # identity filter (keeps any input the same after convolution)
    I = np.zeros_like(G)
    center = kernel_size // 2
    I[center, center] = 1.0

    # unsharp mask kernel 
    unsharp_kernel = (1 + alpha) * I - alpha * G # img + (img - img*G)
    sharpened = convolve2d(img, unsharp_kernel, mode='same', boundary='fill')
    sharpened_clip = np.clip(sharpened, 0, 1)
    return sharpened, sharpened_clip

# ----- Main -----
print("Starting 2.1")

img = img_as_float(io.imread("submission/inputs/2.1_taj.jpg")[..., :3]) # (H,W,3)
img = rgb2gray(img)

sharpened_unclipped, sharpened_clip = unsharp_mask_filter(img, kernel_size=9, gaussian_std=2.0, alpha=1.0)
_, sharpened_half = unsharp_mask_filter(img, kernel_size=9, gaussian_std=2.0, alpha=0.5)

# visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original")
axes[1].imshow(sharpened_half, cmap='gray'); axes[1].set_title("Sharpened a=0.5 (Clipped)")
axes[2].imshow(sharpened_clip, cmap='gray'); axes[2].set_title("Sharpened a=0.1 (Clipped)")
#axes[3].imshow(sharpened_unclipped, cmap='gray'); axes[3].set_title("Sharpened (Unclipped)")
for i in range(3):
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# ========================================================
# Part 2.2: Hybrid Images

def low_pass_filter(im, sigma):
    ksize = int(6*sigma + 1)  # standard heuristic
    G = gaussian_2d(ksize, sigma)
    return convolve2d(im, G, mode="same", boundary="symm")

def high_pass_filter(im, sigma):
    low = low_pass_filter(im, sigma)
    return im - low

def hybrid_image(im1, im2, sigma_high, sigma_low):
    im1_high = high_pass_filter(im1, sigma_high)
    im2_low  = low_pass_filter(im2, sigma_low)
    hybrid = im1_high + im2_low
    hybrid = np.clip(hybrid, 0, 1)  # keep valid range
    return im1_high, im2_low, hybrid

def fourier_spectrum(im):
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(im))))
    return F

# ----- Main -----
print("Starting 2.2")

# load, align and greyscale input images
im1 = img_as_float(plt.imread("submission/inputs/2.2_owl.png")) # high freq
im2 = img_as_float(plt.imread("submission/inputs/2.2_butterfly.png")) # low freq
im1_aligned, im2_aligned = align_images(im1, im2) # align images
im1_aligned_grey, im2_aligned_grey = rgb2gray(im1_aligned), rgb2gray(im2_aligned) # greyscale

# build hybrid
sigma1 = 8   # high-pass: small sigma -> very fine
sigma2 = 12   # low-pass: large sigma -> very blur
im1_high, im2_low, hybrid = hybrid_image(im1_aligned_grey, im2_aligned_grey, sigma1, sigma2)

images = {
    "Input 1": im1,
    "Input 2": im2,
    f"High-Pass": im1_high,
    f"Low-Pass": im2_low,
    "Hybrid": hybrid
}

fouriers = {name: fourier_spectrum(img) for name, img in images.items()}

# visualize intermediates 
fig, axes = plt.subplots(2, len(images), figsize=(4*len(images), 8))
for i, (name, img) in enumerate(images.items()):
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(name)
    axes[0, i].axis('off')
    axes[1, i].imshow(fouriers[name], cmap='gray')
    axes[1, i].set_title(f"Fourier of {name}")
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()

plt.imsave("butterfly-owl.png", hybrid, cmap="gray")

# ========================================================
# Part 2.3: Gaussian and Laplacian Stacks

# Convolution with Gaussian filter
def gaussian_filter(im, sigma):
    ksize = int(6 * sigma + 1)
    G = gaussian_2d(ksize, sigma)
    if im.ndim == 2:  # grayscale
        return convolve2d(im, G, mode="same", boundary="symm")
    elif im.ndim == 3:  # color RGB
        return np.stack([
            convolve2d(im[..., c], G, mode="same", boundary="symm")for c in range(3)
            ], axis=-1)

# Gaussian stack
def gaussian_stack(im, levels, sigma):
    stack = [im]
    current = im
    for i in range(1, levels):
        blurred = gaussian_filter(current, sigma)
        stack.append(blurred)
        current = blurred
    return stack 

# Laplacian stack = Gaussian(level) - Gaussian(level+1)
def laplacian_stack(im, levels, sigma):
    g_stack = gaussian_stack(im, levels, sigma)
    l_stack = []
    for i in range(levels - 1):
        lap = g_stack[i] - g_stack[i + 1]
        l_stack.append(lap)
    l_stack.append(g_stack[-1])  # last level = coarsest Gaussian
    return l_stack 

def normalise_stack(stack):
    norm_stack = []
    for img in stack: 
        norm_img = (img-img.min()) / (img.max()-img.min()+ 1e-8) 
        norm_stack.append(norm_img) 
    return norm_stack

# ----- Main -----
print("Starting 2.3")

im1 = img_as_float(io.imread("submission/inputs/2.3_apple.jpeg")[..., :3]) # (H,W,3)
im2 = img_as_float(io.imread("submission/inputs/2.3_orange.jpeg")[..., :3]) # (H,W,3)

# build gaussian and laplacian stacks
levels = 4
sigma = 2
im1_g_stack = gaussian_stack(im1, levels, sigma)    # list [(H,W,3) * levels]
im1_l_stack = laplacian_stack(im1, levels, sigma)   # list [(H,W,3) * levels]
im1_l_stack_norm = normalise_stack(im1_l_stack)     # list [(H,W,3) * levels]
im2_g_stack = gaussian_stack(im2, levels, sigma)
im2_l_stack = laplacian_stack(im2, levels, sigma)
im2_l_stack_norm = normalise_stack(im2_l_stack)

# visualize
fig, axes = plt.subplots(2, levels, figsize=(15, 6))
for i in range(levels):
    axes[0, i].imshow(im1_g_stack[i], cmap='gray')
    axes[0, i].set_title(f"Apple G{i}")
    axes[0, i].axis("off")
    
    axes[1, i].imshow(im1_l_stack_norm[i], cmap='gray')
    axes[1, i].set_title(f"Apple L{i}")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, levels, figsize=(15, 6))
for i in range(levels):
    axes[0, i].imshow(im2_g_stack[i], cmap='gray')
    axes[0, i].set_title(f"Orange G{i}")
    axes[0, i].axis("off")
    
    axes[1, i].imshow(im2_l_stack_norm[i], cmap='gray')
    axes[1, i].set_title(f"Orange L{i}")
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()

# ========================================================
# Part 2.4: Multiresolution Blending

def reconstruct_from_l_stack(l_stack): 
    img = l_stack[-1] # start from coarsest 
    for i in range(len(l_stack)-2, -1, -1): 
        img = img + l_stack[i]
    return img

# ----- Main -----
print("Starting 2.4")

im1 = img_as_float(io.imread("submission/inputs/2.3_apple.jpeg")[..., :3])  # RGB
im2 = img_as_float(io.imread("submission/inputs/2.3_orange.jpeg")[..., :3])  # RGB
h1, w1, _ = im1.shape 
h2, w2, _ = im2.shape

# center align images and pad im2 if needed
top_pad = (h1 - h2) // 2
bottom_pad = h1 - h2 - top_pad
if top_pad > 0: 
    im2 = np.pad(im2, ((top_pad, bottom_pad), (0,0), (0,0)), mode='constant')

# vertical mask (for orapple & fuji)
mask_vert = np.zeros_like(im1) # (H, W, 3)
mask_vert[:, :(w1//2+70000), :] = 1

# horizontal mask (for egg bald man)
mask_hori = np.zeros_like(im1) # (H, W, 3)
mask_hori[:w1//2, :, :] = 1

# elliptical mask (for penguin in sky)
mask_oval = np.ones((h1, w1, 3))  # outside oval = 1 (show im1)
y, x = np.ogrid[:h1, :w1]
center_y, center_x = h1//2, w1//2
a, b = w1//4, h1//10 # radius: width/4, height/2
ellipse = ((x - center_x)/a)**2 + ((y - center_y)/b)**2 <= 1
mask_oval[ellipse] = 0  # inside oval = 0 (show im2)

# select mask 
mask = mask_vert

# build gaussian and laplacian stacks
levels = 5
sigma = 2
im1_l_stack = laplacian_stack(im1, levels, sigma)   # lists of [(h1,w1,3) * levels]
im2_l_stack = laplacian_stack(im2, levels, sigma)
mask_g_stack = gaussian_stack(mask, levels, 10) 
print({"built stacks"})

blended_stack = []
for l_im1, l_im2, m in zip(im1_l_stack, im2_l_stack, mask_g_stack): 
    blend = (l_im1 * m) + (l_im2 * (1-m))
    blended_stack.append(blend)
print({"blended stack made"})

blended_img = reconstruct_from_l_stack(blended_stack)
blended_img = (blended_img - blended_img.min()) / (blended_img.max() - blended_img.min() + 1e-8)

plt.imshow(blended_img)
plt.axis('off')
plt.show()