import numpy as np

# T = 100
DNA_BLUR = (5, 1.5, 1.5)  # (1, 1, 1) 
DNA_STRICTNESS = 0.45  #0.40
NOISE_PERCENTILE = 50
SIGNAL_PERCENTILE = 92
EROSION1 = np.ones((2, 2), np.uint8)  # y, x
DILATION1 = np.ones((2, 2), np.uint8)  # y, x