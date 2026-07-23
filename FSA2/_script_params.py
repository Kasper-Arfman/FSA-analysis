import numpy as np
import sys
import os
from pyjacket import filetools

uffs_path = os.path.join(os.path.dirname(__file__), '..')
uffs_path = os.path.abspath(uffs_path)
if uffs_path not in sys.path:
    sys.path.insert(0, uffs_path)

TARGET = None  # Only analyze things filenames containing specified string e.g. 'FL'

# == Path settings == # 
DATA_ROOT = r'D:\data'
ANALYSIS_MANUAL = r'C:\Users\arfma005\Documents\GitHub\FSA-analysis\annotations'
ANALYSIS_ROOT = r'C:\Users\arfma005\Documents\GitHub\FSA-analysis\analysis'
CONFIG = filetools.read_yaml(r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\FSA2\config.yaml")

# == 1_annotate == #
PROT_CH_DEFAULT = 0
DNA_CH_DEFAULT = 1


# == 2_extract_rois == #
ROI_LENGTH = {
    'lambda': 248,  # pixels (16 um)
    'dashdot': 62,  # pixels ( 4 um)
}
ROI_MARGIN = (16, 16)  # x, y
DEFAULT_SCALE = [[0, 100], [0, 100]]


# == 3_process_rois == #
CALC_BLUR = 500
DNA_TRESH = 80  #'th percentile
DNA_BLUR = (1, 1, 4)  # t, y, x
PROT_BLUR = (1, 1, 1)
MINOR_BLUR = (1, 1, 1)
CHAIN_BLUR = (5, 5, 5)
DNA_STRICTNESS = 0.40  # higher values leave fewer white pixels
NOISE_PERCENTILE = 50
SIGNAL_PERCENTILE = 92
PROT_STRICTNESS = 0.3
THICKNESS = 80
EROSION1 = np.ones((1, 1), np.uint8)  # y, x
DILATION1 = np.ones((2, 2), np.uint8)  # y, x
TI_DEFAULT = 0  # 75
TF_DEFAULT = -1  # -50

if __name__ == "__main__":
    print('\nScript_params is a configuration file, do not run it directly.')