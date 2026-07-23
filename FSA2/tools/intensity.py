import numpy as np
from pyjacket import arrtools, filetools

def relative_index(sizes, i):
    for j, size in enumerate(sizes):
        if i < size:
            return j, i   
        i -= size
    raise IndexError()

def percentile(hist: np.ndarray, p, color=False) -> np.ndarray:
    if color:
        hist = np.cumsum(hist, axis=0)
        i = np.stack(
            [np.searchsorted(hist[:, i], p, side='right') \
                for i in range(hist.shape[-1])]).T
    else:
        hist = np.cumsum(hist)
        i = np.searchsorted(hist, p, side='right')
    return i.T

def intensity_histogram(a: filetools.ImageHandle, color=True, normalize=True) -> np.ndarray:
    """Compute the histogram for a lazy generator of numpy arrays"""

    if color:
        shape = (arrtools.type_max(a.dtype)+1, a.shape[-1])  # e.g. (256, 3)
        hist = np.zeros(shape, dtype=np.int64)
        for rgb in a:
            for i in range(shape[-1]):
                frame = rgb[..., i]
                unique, counts = np.unique(frame, return_counts=True)
                hist[unique, i] += counts
        if normalize:  hist = hist / np.sum(hist, axis=0)
                
    else:
        shape = arrtools.type_max(a.dtype)+1  # e.g. (256, )
        hist = np.zeros(shape, dtype=np.int64)
        for frame in a:
            unique, counts = np.unique(frame, return_counts=True)
            hist[unique] += counts
        if normalize:  hist = hist / np.sum(hist)
        
    return hist 