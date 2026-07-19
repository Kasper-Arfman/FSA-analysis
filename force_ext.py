import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np
import cv2
from pyjacket import filetools, plottools
plottools.style.use('Sprakel')

T = 100
DNA_BLUR = (0, 1.5, 1.5)  # (1, 1, 1) 
DNA_STRICTNESS = 0.45  #0.40
NOISE_PERCENTILE = 50
SIGNAL_PERCENTILE = 92
EROSION1 = np.ones((2, 2), np.uint8)  # y, x
DILATION1 = np.ones((2, 2), np.uint8)  # y, x

def isolate(img, contour):
    """Sets intensities outside contour to zero"""
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(img, mask)

def isolate_largest(dna_channel):
    out = np.zeros_like(dna_channel)
    for i, frame in enumerate(dna_channel):
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            frame = isolate(frame, largest_contour)

        out[i] = frame

    return out

def measure_length_bead(path: str):
    img = tifffile.imread(path)

    # 3D gaussian blur
    img_blurred = gaussian_filter(img, sigma=DNA_BLUR)

    filetools.write_img('test.tif', img_blurred)

    # > Set threshold levels
    dna_threshold = sum([
        (1-DNA_STRICTNESS)*np.percentile(img_blurred[DNA_BLUR[0]], NOISE_PERCENTILE), 
        (  DNA_STRICTNESS)*np.percentile(img_blurred, SIGNAL_PERCENTILE)])

    # > Binarize
    i_dna = np.zeros_like(img_blurred)
    i_dna[img_blurred > dna_threshold] = 255
    i_dna = np.astype(i_dna, np.uint8)

    # > isolate only the maximal contour
    i_dna = isolate_largest(i_dna)

    filetools.write_img('i_test.tif', i_dna)

    img_blurred[i_dna == 0] = 0

    contact_points = []
    bead_peaks = []

    for i, (frame0, img0) in enumerate(zip(i_dna, img_blurred)):
        line = frame0.sum(axis=0)  # number of white pixels along x
        line2 = np.sum(img0, axis=0)

        pixel_indices = np.argwhere(line > 0)
        bead_peak = np.argmax(line2)
        contact_point = pixel_indices[0]

        contact_points.append(contact_point)
        bead_peaks.append(bead_peak)
    bead_peaks = np.array(bead_peaks)

    contact_point = np.median(contact_points[-50:])
    lengths = bead_peaks - contact_point
    return lengths



def measure_length_free(path: str):
    img = tifffile.imread(path)

    # 3D gaussian blur
    img_blurred = gaussian_filter(img, sigma=DNA_BLUR)

    filetools.write_img('test.tif', img_blurred)



    # > Set threshold levels
    dna_threshold = sum([
        (1-DNA_STRICTNESS)*np.percentile(img_blurred[DNA_BLUR[0]], NOISE_PERCENTILE), 
        (  DNA_STRICTNESS)*np.percentile(img_blurred, SIGNAL_PERCENTILE)])

    # > Binarize
    i_dna = np.zeros_like(img_blurred)
    i_dna[img_blurred > dna_threshold] = 255
    i_dna = np.astype(i_dna, np.uint8)

    # > isolate only the maximal contour
    i_dna = isolate_largest(i_dna)

    filetools.write_img('i_test.tif', i_dna)

    img_blurred[i_dna == 0] = 0

    contact_points = []
    bead_peaks = []

    for i, (frame0, img0) in enumerate(zip(i_dna, img_blurred)):
        line = frame0.sum(axis=0)  # number of white pixels along x
        # line2 = np.sum(img0, axis=0)

        pixel_indices = np.argwhere(line > 0)
        bead_peak = pixel_indices[-1]  ##np.argmax(line)
        contact_point = pixel_indices[0]

        contact_points.append(contact_point)
        bead_peaks.append(bead_peak)
    bead_peaks = np.array(bead_peaks)

    contact_point = np.median(contact_points[-50:])
    lengths = bead_peaks - contact_point
    return lengths


if __name__ == "__main__":
    BEAD_RADIUS = 810e-9  / 2
    MPP = 6.5e-6 / 100
    FPS = 10

    paths = [
        (r"data\Force extension 0-12.93 mbar_62x52y_free.tif", 0, None),
        (r"data\Force extension 0-12.93 mbar_112x82y_free.tif", 0, None),
        (r"data\Force extension 0-12.93 mbar_146x145y_free.tif", 0, None),
        (r"data\Force extension 0-12.93 mbar_170x200y_free.tif", 0, None),
    ]

    paths += [
        (r"data\Force extension 0-12.93 mbar_155x111y_bead.tif", 0, None),
        (r"data\Force extension 0-12.93 mbar_157x132y_bead.tif", 0, None),
        (r"data\Force extension 0-12.93 mbar_163x124y_bead.tif", 0, None),
    ]

    
    for path, ti, tf in paths:
        if 'bead' in path:
            lengths = measure_length_bead(path)
            lengths *= MPP
            lengths -= BEAD_RADIUS
        else:
            lengths = measure_length_free(path)
            lengths *= MPP

        lengths = median_filter(lengths, size=10)
        
        tf = tf or len(lengths)
        lengths = lengths[ti:tf]


        plt.plot(np.arange(ti, tf) / FPS, lengths * 1e6)

    plt.axhline(16.5, color='k', linestyle='--')
    plt.ylabel('Length ($\mu m$)')
    plt.xlabel('Time ($s$)')
    plt.xlim([0, None])
    plt.ylim([0, None])
    plt.show()