import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np
import cv2
from scipy.constants import k
from pyjacket import filetools, plottools
plottools.style.use('Sprakel')

ETA = 1e-3
R = 405e-9
L = 16.5e-6

f_bead = 6 * np.pi * ETA * R

def F_WLC(x, T=293, Lp=50e-9):
    return k*T * ((1-x)**-2 + 4*x - 1) / (4 * Lp)
    
Lb = 15.65e-6
Lf = 14.05e-6

Lb = 12.90e-6
Lf =  4.75e-6




Ff = F_WLC(Lf/L)
Fb = F_WLC(Lb/L)
print(f"Free: {Ff*1e12:.2f} pN\nBead: {Fb*1e12:.2f} pN")

v = (Fb - Ff) / f_bead  # m/s
print(f"v = {v*1e6:.2f} um/s")

f_DNA = Ff / v
print(f"f_DNA = {f_DNA*1e9:.2f}, f_bead = {f_bead*1e9:.2f}, ({100*f_DNA/f_bead:.1f}%)")


# The longer the DNA, the stronger the compacting force of the protein




