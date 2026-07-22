from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

N = 30

def pixelize_image(image_path, target_width=N, target_height=N):
    """
    Convert an image to grayscale and pixelize it to a target resolution.
    
    Parameters:
        image_path (str): Path to the input image.
        target_width (int): Desired width of the pixelated image.
        target_height (int): Desired height of the pixelated image.
    
    Returns:
        np.ndarray: 2D array of grayscale pixel values (0-255).
    """
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')  # 'L' mode is grayscale
    
    # Resize image to target resolution
    img_small = img.resize((target_width, target_height), resample=Image.BILINEAR)
    
    # Convert to numpy array
    pixels = np.array(img_small)
    
    return pixels


def gaussian_kernel(size=5, sigma=1.0):
    """Generate a 2D Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)




q = pixelize_image('pixelduck.png')


kernel_5x5 = gaussian_kernel(size=5, sigma=1.0)

# print(kernel_5x5)

k = 255 * kernel_5x5 / np.max(kernel_5x5)

for v in k:
    print(' '.join(f'{val:.4f}' for val in v))

exit()




# Perform convolution
q = convolve2d(q, kernel_5x5, mode='same', boundary='symm')




# print(q)

for row in q:
    print(' '.join(f'{val:.0f}' for val in row))

plt.imshow(q, cmap='gray', vmin=0, vmax=255)
plt.show()