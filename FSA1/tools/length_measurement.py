import tifffile
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
from _script_params import *

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

def measure_length_free(path: str, sigma=DNA_BLUR, P_noise=NOISE_PERCENTILE, P_signal=SIGNAL_PERCENTILE, strictness=0.45):
    img = tifffile.imread(path)

    img_blurred = gaussian_filter(img, sigma=sigma)
    dna_threshold = sum([
        (1-strictness)*np.percentile(img_blurred[sigma[0]], P_noise), 
        (  strictness)*np.percentile(img_blurred, P_signal)])

    # > Binarize
    i_dna = np.zeros_like(img_blurred)
    i_dna[img_blurred > dna_threshold] = 255
    i_dna = np.astype(i_dna, np.uint8)

    # > isolate only the maximal contour
    i_dna = isolate_largest(i_dna)

    img_blurred[i_dna == 0] = 0

    contact_points = []
    bead_peaks = []
    for i, (frame0, img0) in enumerate(zip(i_dna, img_blurred)):
        line = frame0.sum(axis=0)  # number of white pixels along x
        pixel_indices = np.argwhere(line > 0)
        bead_peak = pixel_indices[-1]  ##np.argmax(line)
        contact_point = pixel_indices[0]

        contact_points.append(contact_point)
        bead_peaks.append(bead_peak)
    bead_peaks = np.array(bead_peaks)

    contact_point = np.median(contact_points[-50:])
    lengths = bead_peaks - contact_point
    return lengths

def measure_length_bead(path: str, sigma=DNA_BLUR, P_noise=NOISE_PERCENTILE, P_signal=SIGNAL_PERCENTILE, strictness=0.45):
    img = tifffile.imread(path)

    # 3D gaussian blur
    img_blurred = gaussian_filter(img, sigma=sigma)

    # > Set threshold levels
    dna_threshold = sum([
        (1-strictness)*np.percentile(img_blurred[sigma[0]], P_noise), 
        (  strictness)*np.percentile(img_blurred, P_signal)])

    # > Binarize
    i_dna = np.zeros_like(img_blurred)
    i_dna[img_blurred > dna_threshold] = 255
    i_dna = np.astype(i_dna, np.uint8)

    # > isolate only the maximal contour
    i_dna = isolate_largest(i_dna)

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