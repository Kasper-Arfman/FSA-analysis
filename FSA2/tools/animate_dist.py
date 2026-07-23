import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

mpl.rcParams['animation.ffmpeg_path'] = r'C:\<SPECIFY_FFMPEG_PATH_HERE>'
green = LinearSegmentedColormap.from_list("black_to_green", [(0, 0, 0), (0, 1, 0)])
magenta = LinearSegmentedColormap.from_list("black_to_magenta", [(0, 0, 0), (1, 0, 1)])

def animate_distribution(roi: np.ndarray, kym: np.ndarray, frame_step=1, c='white'):
    roi = roi[::frame_step]
    roi_min = np.min(roi, axis=(0, 1, 2))
    roi_max = np.max(roi, axis=(0, 1, 2))
    kym = kym[::frame_step]
    kymin = np.min(kym, axis=(0, 1))
    kymax = np.max(kym, axis=(0, 1))
    kymod =  roi.shape[1] * (1 - (kym - kymin) / (kymax - kymin))

    # Initialize figure
    ax0: Axes
    ax1: Axes
    fig, (ax0, ax1) = plt.subplots(2, 1)

    im0 = ax0.imshow(roi[0, ..., 0], vmin=roi_min[0], vmax=roi_max[0], cmap=magenta)
    l0, = ax0.plot([], [], c)
    im1 = ax1.imshow(roi[0, ..., 1], vmin=roi_min[1], vmax=roi_max[1], cmap=green)
    l1, = ax1.plot([], [], c)
    for ax in [ax0, ax1]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.patch.set_facecolor('black')      # Figure background

    kym_x = np.arange(kym.shape[1])
    def update(frame: int):
        im0.set_data(roi[frame, ..., 0])
        im1.set_data(roi[frame, ..., 1])
        l0.set_data(kym_x, kymod[frame, ..., 0])
        l1.set_data(kym_x, kymod[frame, ..., 1])
        return [im0, l0, im1, l1]

    # Create animation
    frames = np.arange(0, roi.shape[0])
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    plt.close(fig)
    return ani

def animate_distribution(roi: np.ndarray, c='white', prot_ch=0):
    kym: np.ndarray = np.mean(roi, axis=1)
    kymin = np.min(kym, axis=(0, 1))
    kymax = np.max(kym, axis=(0, 1))
    kymod: np.ndarray =  roi.shape[1] * (1 - (kym - kymin) / (kymax - kymin + 1e-8))

    divisors = np.arange(1, kymod.shape[0] + 1).reshape(-1, 1)
    kymod_cum = np.cumsum(kymod[..., prot_ch], axis=0) / divisors

    # ROI
    roi_min = np.min(roi, axis=(0, 1, 2))
    roi_max = np.max(roi, axis=(0, 1, 2))
    roi = 255 * (roi.astype(np.float32) - roi_min) / (roi_max - roi_min + 1e-8)
    roi = roi.astype(np.uint8)
    roi = np.stack([roi[..., 0], roi[..., 1], roi[..., 0]], axis=-1)

    # Initialize figure
    fig, ax = plt.subplots(1, 1)
    im0 = ax.imshow(roi[0], vmin=0, vmax=255)
    l0, = ax.plot([], [], c)
    l1, = ax.plot([], [], 'blue')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.patch.set_facecolor('black')

    kym_x = np.arange(kym.shape[1])
    def update(frame: int):
        im0.set_data(roi[frame])
        l0.set_data(kym_x, kymod[frame, ..., prot_ch])
        l1.set_data(kym_x, kymod_cum[frame])
        return [im0, l0]

    # Create animation
    frames = np.arange(0, roi.shape[0])
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    plt.close(fig)
    return ani