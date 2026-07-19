from PIL import Image, ImageSequence
import cv2
import numpy as np
import matplotlib.pyplot as plt

# PIXEL_SIZE = 16/200  # [um / pixel]
PIXEL_SIZE = 6.5 / 100
BLUR_SIZE = (5, 5)
MEDIAN_THRESHOLD = 91  # /255
BINARY_THRESHOLD = 11  # /255
FRAME_TIME = 500  # [ms]


def median_filter(image, size, subtract=False, **kwargs):
    background = cv2.medianBlur(image, size, **kwargs)
    if subtract:
        return subtract_uint(image, background)
    else:
        return background
    
def subtract_uint(a, b):
    return np.where(a > b, a-b, 0)

def imread_tif(filepath, as_gray=False):
    # Read an RGB .tiff file
    movie = Image.open(filepath)
    result = []
    for frame in ImageSequence.Iterator(movie):
        
        if as_gray:
            frame = frame.convert('L')
            
        result.append(frame)
    movie = np.array(result)
    return movie

def measure_strand_length(gray):
    # Blur image to reduce noise
    blurred = cv2.GaussianBlur(gray, BLUR_SIZE, cv2.BORDER_DEFAULT)
    
    # Remove background by subtracting the local median
    bkgr = median_filter(blurred, MEDIAN_THRESHOLD, subtract=True)
    
    # Convert into a binary image
    _, thresh = cv2.threshold(bkgr, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Dilate and erode to repair small gaps in the DNA strand
    if True:
        src = thresh
        shape = cv2.MORPH_CROSS
        size = 4
        element = cv2.getStructuringElement(shape, (2*size+1, 2*size+1), (size, size))
        src = cv2.dilate(src, element)
        src = cv2.erode(src, element)
        thresh = src

    # Assume the largest white 'blob' is the DNA, and compute length in x-axis
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:  return 0   
    contour = max(contours, key=cv2.contourArea)
    x = contour[:, 0, 0]
    length = (max(x) - min(x)) * PIXEL_SIZE
    return thresh, length

def process_tiff(movie):
    strand_lengths = []
    for frame in movie:
        frame, length = measure_strand_length(frame)
        strand_lengths.append(length)
        cv2.imshow('processed', frame)
        cv2.waitKey(1)
    return strand_lengths

def main(filename):
    movie = imread_tif(filename, as_gray=True)
    N = len(movie)
    times = np.arange(N) * FRAME_TIME/1000
    
    strand_lengths = process_tiff(movie)
    
    print("Strand lengths for each frame:")
    for i, (time, length) in enumerate(zip(times, strand_lengths)):
        print(f"{i};{time:.2f};{length:.4f}")
        
    plt.plot(times, strand_lengths)
    plt.xlabel('time [s]')
    plt.ylabel('length [um]')
    plt.show()

if __name__ == "__main__":
    
    filename = [
        'test.tif',
        'bead001.tif',
        'bead002.tif',
        "ARF dil1 10nM_85x88y_bead.tif" ,
        'ARF dil1 10nM_14x58y_bead.tif',
        "ARF dil1 10nM_44x66y_free.tif",
        "ARF dil1 10nM_65x38y_free.tif",
        "ARF dil1 10nM_68x101y_free.tif",
        "ARF dil1 10nM_98x61y_free.tif",
        "ARF dil1 10nM_111x95y_bead.tif",
        'ARF dil1 10nM_124x82y_free.tif',
        "ARF dil1 10nM_126x95y_bead.tif",
        "ARF dil1 10nM_146x75y_bead.tif",
        "Force extension 0-12.93 mbar_62x52y_free.tif",
        "Force extension 0-12.93 mbar_112x82y_free.tif",
        "Force extension 0-12.93 mbar_146x145y_free.tif",
        "Force extension 0-12.93 mbar_155x111y_bead.tif",
        "Force extension 0-12.93 mbar_157x132y_bead.tif",
        "Force extension 0-12.93 mbar_163x124y_bead.tif",
        "Force extension 0-12.93 mbar_170x200y_free.tif",
        "Force extension_ramp001_115x48y_bead.tif",
        "Force extension_ramp001_146x70y_bead.tif",
        "Force extension_ramp001_59x127y_bead.tif",
        "Force extension_ramp001_37x68y_free.tif",
        "Force extension_ramp001_41x84y_free.tif",
        "Force extension_ramp001_72x79y_free.tif",
        "bead.tif"
    ][26]
    
    
    main(filename=filename)
