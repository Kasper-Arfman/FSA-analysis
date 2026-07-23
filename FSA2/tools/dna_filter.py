import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def isolate(img, contour):
    """Sets intensities outside contour to zero"""
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(img, mask)

def clean_roi(roi: np.ndarray, i_dna:np.ndarray, prot_ch=0, dna_ch=1):
    """Remove signal from outside i_dna"""
    dna = roi[..., dna_ch] * i_dna
    prot = roi[..., prot_ch] * i_dna

    roi = np.empty_like(roi, dtype=roi.dtype)
    roi[..., dna_ch] = dna
    roi[..., prot_ch] = prot
    return roi

def track_DNA(dna_channel, rel_thresh=0.5, blur_size=(2, 2, 2)):
    dna_channel = gaussian_filter(dna_channel, sigma=blur_size)
    return threshold_DNA(dna_channel, rel_thresh)

def threshold_DNA(dna_channel, noise_level):
    for i, frame in enumerate(dna_channel):
        frame[frame <= noise_level] = 0
        frame[frame  > noise_level] = 255
        dna_channel[i] = frame
    return dna_channel

def isolate_largest(dna_channel):
    out = np.zeros_like(dna_channel)
    for i, frame in enumerate(dna_channel):
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            frame = isolate(frame, largest_contour)
        out[i] = frame
    return out