from pyjacket import filetools, plottools
import tifffile
import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
# plottools.style.use('Sprakel')

a = -28.6
def rotate_frame(frame: np.ndarray, degrees: float, **kwargs) -> np.ndarray:
    kwargs.setdefault('reshape', False)
    return ndimage.rotate(frame, -degrees, **kwargs)

ROOT = r'D:\MiCube\2023-11-24_FL_Lambda_KCl'
file = '10nM_stab_1_roi1'
path = rf"{ROOT}\{file}.tif"

# img = filetools.read_img(f'{path}\\{file}.tif')

Y, X = 144, 111
L = 150
M = 15
MED = 51
T0, T1 = 0, 84
FRAMES = [
    0, 
    39, 
    49,
    52, 
    70]
sz = 1.5
sxy = 2.5

# == Read image ==
img = tifffile.memmap(path)[T0:T1]
img = img.astype(np.float32)
print('Read image shape:', img.shape)

# #  - Background correction
background_img = np.median(img, axis=0)
background = median_filter(background_img, size=51)
img = img - background
img[img < 0] = 0
gamma = 0.6

#  - Signal enhancement
img = gaussian_filter(img, sigma=(sz, sxy, sxy))

# == Visualization ==
vmin, vmax = np.percentile(img[:, Y-M:Y+M, X-M:X+L+M], [50, 99.5])
vmax = 400

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

green = LinearSegmentedColormap.from_list(
    "black_to_green",
    ["black", "green"]
)

fig, ax = plt.subplots(nrows=len(FRAMES), ncols=1, gridspec_kw={'wspace':0, 'hspace':0.01})

for i, f in enumerate(FRAMES):
    frame = img[f, Y-M:Y+M, X-M:X+L+M]


    


    frame = frame**gamma *  vmax/(vmax**gamma)

    vmin = np.min(frame)
    vmax = np.max(frame)

    # vmax = np.percentile(frame, 99.9)
    # vmax = np.max(frame)

    ax[i].imshow(frame, cmap=green, vmin=vmin, vmax=vmax)
    ax[i].set_axis_off()

# plt.subplots_adjust(
#     left=0, right=1, top=1, bottom=0, hspace=0, wspace=0
# )
plt.show()