import numpy as np
from skimage import io, img_as_float, exposure
import matplotlib.pyplot as plt

# -------------------------------------------------
# Function: align one channel to a reference channel using SSD
def align(channel, ref, max_shift=15):
    """
    Align `channel` to `ref` by searching over [-max_shift, max_shift] pixels in x and y.
    Uses SSD (sum of squared differences).
    """
    best_displacement = (0, 0)
    min_error = np.inf
    crop = 15  # avoid borders
    h, w = ref.shape
    
    # [for gradient based alignment] compute edges instead of using raw intensities 
    # ref_edges = sobel(ref)
    # chan_edges = sobel(channel)

    for dx in range(-max_shift, max_shift+1):
        for dy in range(-max_shift, max_shift+1):
            shifted = np.roll(np.roll(channel, dx, axis=0), dy, axis=1)
            
            # compute SSD error using intensities 
            error = np.sum((ref[crop:h-crop, crop:w-crop] - shifted[crop:h-crop, crop:w-crop])**2)

            # [for gradient based alignment] compute SSD error using edges 
            # error = np.sum((ref_edges[crop:h-crop, crop:w-crop] -shifted[crop:h-crop, crop:w-crop])**2)
            
            if error < min_error:
                min_error = error
                best_displacement = (dx, dy)
        
    # apply best displacement
    dx, dy = best_displacement
    aligned = np.roll(np.roll(channel, dx, axis=0), dy, axis=1)

    return aligned, best_displacement

# -------------------------------------------------
# Function: make image pyramid
def align_pyramid(channel, ref, max_shift=15, levels=3):
    if levels == 0 or min(channel.shape) < 50:
        return align(channel, ref, max_shift)
    
    # downsample (by factor of 2, take every 2 pixels)
    channel_small = channel[::2, ::2]
    ref_small = ref[::2, ::2]
    
    # recursive alignment
    _, offset_small = align_pyramid(channel_small, ref_small, max_shift, levels-1)
    
    # scale offset to current resolution
    dx, dy = offset_small
    dx *= 2
    dy *= 2
    
    # refine locally around predicted shift
    search_range = 3
    best_offset = (dx, dy)
    min_error = np.inf
    h, w = ref.shape
    crop = 15
    
    for delta_x in range(-search_range, search_range+1):
        for delta_y in range(-search_range, search_range+1):
            total_dx = dx + delta_x
            total_dy = dy + delta_y
            shifted = np.roll(np.roll(channel, total_dx, axis=0), total_dy, axis=1)
            error = np.sum((ref[crop:h-crop, crop:w-crop] - shifted[crop:h-crop, crop:w-crop])**2)
            
            if error < min_error:
                min_error = error
                best_offset = (total_dx, total_dy)
    
    final_aligned = np.roll(np.roll(channel, best_offset[0], axis=0),
                            best_offset[1], axis=1)
    
    return final_aligned, best_offset

# -------------------------------------------------
# Main 
imname = 'raw_data/extra_factory.tiff'
im = img_as_float(io.imread(imname))

# split stacked channels (top to bottom: B, G, R)
height = im.shape[0] // 3
b = im[:height, :]
g = im[height:2*height, :]
r = im[2*height:3*height, :]

# align channels to blue
if imname.lower().endswith(('.tif', '.tiff')):
    g_aligned, g_disp = align_pyramid(g, b, max_shift=30, levels=3)
    r_aligned, r_disp = align_pyramid(r, b, max_shift=30, levels=3)
else:  # assume .jpg or other smaller image formats
    g_aligned, g_disp = align(g, b)
    r_aligned, r_disp = align(r, b)

# print displacements
print("\n" + "="*50)
print(f" Image: {imname}")
print(f" G displacement: {g_disp}")
print(f" R displacement: {r_disp}")
print("="*50 + "\n")

# stack into color image
im_out = np.dstack([r_aligned, g_aligned, b])

# visualise using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(im_out)
plt.title(f'Aligned Image: {imname}')
plt.axis('off')
plt.show()

# -------------------------------------------------
# Extra: For making high contrast images

# make contrasted image
contrast_im = exposure.equalize_hist(im_out)

# show original and contrasted image side by side
fig, axes = plt.subplots(1, 2, figsize=(12,6))
axes[0].imshow(im_out)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(contrast_im)
axes[1].set_title("Equalized Contrast")
axes[1].axis("off")
plt.show()
