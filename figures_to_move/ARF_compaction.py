# from pyjacket import filetools, plottools
import tifffile
import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

green = LinearSegmentedColormap.from_list(
    "black_to_green",
    ["black", "green"]
)

ROOT = r'D:\MiCube\2023-11-24_FL_Lambda_KCl'
file = 'assay_buffer_2_roi1'

VMAX = None,  ##400
Y, X = 231, 167
L = 150
M = 15
MED = 51
T0, T1 = 225, 896
FRAMES = [
    # 408,
    503,
    572,
    664,
    820, 
    917,
    ]
sz = 1.5
sxy = 2.5
gamma = 0.6

vmins = np.linspace(0, 0, len(FRAMES))
vmaxs = np.linspace(270, 870, len(FRAMES))

# == Read image ==
img = tifffile.memmap(rf"{ROOT}\{file}.tif")
img = img.astype(np.float32)
print('Read image shape:', img.shape)

#  - Background correction
background_img = np.median(img, axis=0)
background = median_filter(background_img, size=51)
img = img - background
img[img < 0] = 0

# #  - Signal enhancement
img = gaussian_filter(img, sigma=(sz, sxy, sxy))

# == Visualization ==
vmin, vmax = np.percentile(img[:, Y-M:Y+M, X-M:X+L+M], [50, 99.5])
if VMAX: vmax = VMAX
fig, ax = plt.subplots(nrows=len(FRAMES), ncols=1, gridspec_kw={'wspace':0, 'hspace':0.01})




for i, (f, vmin, vmax) in enumerate(zip(FRAMES, vmins, vmaxs)):
    frame = img[f, Y-M:Y+M, X-M:X+L+M]

    # vmin = np.min(frame)
    # vmax = np.max(frame)
    # print(vmin, vmax)

    frame = frame**gamma *  vmax/(vmax**gamma)

    # vmin = np.min(frame)
    # vmax = np.max(frame)
    # print(vmin, vmax)

    ax[i].imshow(frame, cmap=green, vmin=vmin, vmax=vmax)
    ax[i].set_axis_off()
plt.show()