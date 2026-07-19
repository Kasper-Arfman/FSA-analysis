import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve, median_filter
from skimage.morphology import skeletonize
import numpy as np
import cv2
import networkx as nx
from scipy.spatial import distance
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import distance
import networkx as nx
from scipy.constants import k
from pyjacket import filetools, plottools
# plottools.style.use('Sprakel')

plt.style.use(r"C:\Users\arfma005\Documents\GitHub\pyjacket\pyjacket\plottools\styles\sprakel2.mplstyle")

# NEIGHBOR_OFFSETS = [(-1, -1), (-1,  0), (-1,  1), 
#                     ( 0, -1),           ( 0,  1), 
#                     ( 1, -1), ( 1,  0), ( 1,  1)]

# def graph_from_grid(skeleton: np.ndarray):
#     """Convert a binary image into a graph
    
#     """
#     assert skeleton.dtype == np.uint8, ValueError('Skeleton must be np.uint8')
#     pixel_positions = set(zip(*np.where(skeleton > 0)))
    
#     graph = nx.Graph()
#     for pos in pixel_positions:
#         for neighbor in neighbor_positions(*pos, filter=pixel_positions):
#             graph.add_edge(pos, neighbor, weight=distance.euclidean(pos, neighbor))
#     return graph

# def neighbor_positions(r, c, filter=None):
#     """Find all neighboring coordinates.
#     Optionally, return only coordinates that are present in the filter
#     """
#     for dr, dc in NEIGHBOR_OFFSETS:
#         neighbor = (r + dr, c + dc)
#         if filter is not None and neighbor not in filter:
#             continue
#         yield neighbor


# def max_shortest_path(graph: nx.Graph, endpoints: list[tuple]):
#     """Find the pair of endpoints whose shortest path is maximal and return it as a binary image."""
#     endpoints = [tuple(yx) for yx in endpoints]
    
#     pair, path, path_length = (None, [], -float('inf'))
#     for i, p1 in enumerate(endpoints):
#         for p2 in endpoints[i+1:]:
#             try:
#                 pair = (p1, p2)
#                 pth = nx.shortest_path(graph, source=p1, target=p2, weight='weight')
#                 length = sum(distance.euclidean(pth[k], pth[k+1]) for k in range(len(pth)-1))
#                 if length > path_length:
#                     pair, path, path_length = (pair, pth, length)
#             except nx.NetworkXNoPath:
#                 continue
#     return pair, path, path_length

# BRANCH_PATTERNS = np.array([
#     [[0, 1, 0],
#      [1, 1, 1],
#      [0, 0, 0]],

#     [[1, 0, 1,],
#      [0, 1, 0,],
#      [1, 0, 0,]],

#     [[1, 0, 1],
#      [0, 1, 0],
#      [0, 1, 0]],

#     [[0, 1, 0],
#      [1, 1, 0],
#      [0, 0, 1]],

#     [[0, 0, 1],
#      [1, 1, 1],
#      [0, 1, 0]],

#     [[1, 0, 0],
#      [1, 1, 1],
#      [0, 1, 0]],

#     [[0, 1, 0],
#      [1, 1, 0],
#      [0, 1, 0]],

#     [[0, 0, 0],
#      [1, 1, 1],
#      [0, 1, 0]],

#     [[0, 1, 0],
#      [0, 1, 1],
#      [0, 1, 0]],

#     [[1, 0, 0],
#      [0, 1, 0],
#      [1, 0, 1]],

#     [[0, 0, 1],
#      [0, 1, 0],
#      [1, 0, 1]],

#     [[1, 0, 1],
#      [0, 1, 0],
#      [0, 0, 1]],

#     [[1, 0, 0],
#      [0, 1, 1],
#      [1, 0, 0]],

#     [[0, 1, 0],
#      [0, 1, 0],
#      [1, 0, 1]],

#     [[0, 0, 1],
#      [1, 1, 0],
#      [0, 0, 1]],

#     [[0, 0, 1],
#      [1, 1, 0],
#      [0, 1, 0]],

#     [[1, 0, 0],
#      [0, 1, 1],
#      [0, 1, 0]],

#     [[0, 1, 0],
#      [0, 1, 1],
#      [1, 0, 0]],

#     [[1, 1, 0],
#      [0, 1, 1],
#      [0, 1, 0]],

#     [[0, 1, 0],
#      [1, 1, 1],
#      [1, 0, 0]],

#     [[0, 1, 0],
#      [1, 1, 0],
#      [0, 1, 1]],

#     [[0, 1, 0],
#      [0, 1, 1],
#      [1, 1, 0]],

#     [[0, 1, 0],
#      [1, 1, 1],
#      [0, 0, 1]],

#     [[0, 1, 1],
#      [1, 1, 0],
#      [0, 1, 0]],

#     [[0, 1, 0],
#      [1, 1, 1],
#      [0, 1, 0]],

#     [[1, 0, 1],
#      [0, 1, 0],
#      [1, 0, 1]],
    
#     ])

# NEIGHBOR_KERNEL = np.array([[1, 1, 1], 
#                             [1, 0, 1], 
#                             [1, 1, 1]])

# def is_branch_point(skeleton: np.ndarray, y, x):

#     padded_region = np.zeros((3, 3), dtype=skeleton.dtype)


#     # Get the valid region indices
#     y_start, y_end = max(0, y-1), min(skeleton.shape[0], y+2)
#     x_start, x_end = max(0, x-1), min(skeleton.shape[1], x+2)

#     # Compute the corresponding indices in the 3x3 patch
#     patch_y_start, patch_y_end = 1 - (y - y_start), 1 + (y_end - y)
#     patch_x_start, patch_x_end = 1 - (x - x_start), 1 + (x_end - x)

#     # Copy valid data from skeleton into the 3x3 region
#     padded_region[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = skeleton[y_start:y_end, x_start:x_end]


#     return any((
#         np.all(pattern==padded_region) \
#             for pattern in BRANCH_PATTERNS
#     ))



# def critical_points(skeleton: np.ndarray):
#     """Finds end points and branch points for a skeleton image"""
#     assert skeleton.dtype == np.uint8, ValueError('Skeleton must be np.uint8')
    
#     neighbor_count = convolve(skeleton, NEIGHBOR_KERNEL, mode="constant", cval=0)

#     end_points = np.argwhere(skeleton & (neighbor_count == 1))

#     branch_candidates = np.argwhere(skeleton & (neighbor_count >= 3))
#     branch_points = [yx for yx in branch_candidates if is_branch_point(skeleton, *yx)]
#     return end_points, branch_points

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

T = 100
DNA_BLUR = (5, 1.5, 1.5)  # (1, 1, 1) 
DNA_STRICTNESS = 0.45  #0.40
NOISE_PERCENTILE = 50
SIGNAL_PERCENTILE = 92
EROSION1 = np.ones((2, 2), np.uint8)  # y, x
DILATION1 = np.ones((2, 2), np.uint8)  # y, x


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

def wlc_force(r, L, Lp):
    x = r / L
    if x > 1:
        x = 0.95
        print('WARNING, OVERSTRETCHED POLYMER')
    
    # return k*T/(4*Lp) * ((1-x)**-2 + 4*x - 1)  # Approximate form (15% error)
    return k*T/(4*Lp) * ((1-x)**-2 + 4*x - 1 - 3.2*x**2.15)   # 1% error

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

    plt.figure(figsize=(5, 3))
    plt.subplots_adjust(left=0.55/5, right=0.99, top=0.99, bottom=0.45/3)

    paths = [
        (r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\ARF dil1 10nM_14x58y_bead.tif", 10, None),
        (r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\ARF dil1 10nM_85x88y_bead.tif", 0, None),
        (r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\ARF dil1 10nM_146x75y_bead.tif", 0, 463),
    ]

    all_lengths = []
    for path, ti, tf in paths:
        lengths = measure_length_bead(path)
        lengths *= MPP
        lengths -= BEAD_RADIUS

        tf = tf or len(lengths)
        lengths[:ti] = np.nan
        lengths[tf:] = np.nan

        all_lengths.append(lengths.copy())


        plt.plot(np.arange(lengths.shape[0]) / FPS, lengths * 1e6, 'grey')
    all_lengths = np.array(all_lengths)

    ti = 10
    tf = int(30 * FPS)

    lengths_avg = np.nanmean(all_lengths, axis=0)
    avg_length = np.average(lengths_avg[ti:tf]) * 1e6
    print(f"{avg_length = :.2f}")

    F = wlc_force(avg_length * 1e-6, L=16.5e-6, Lp=50e-9)
    print(f"{F*1e12 = :.2f} pN")


    plt.plot(
        np.arange(ti, len(lengths_avg))/FPS, 
        lengths_avg[ti:] * 1e6,
        label='Bead'
        )
    plt.axhline(avg_length, color='C0', linestyle='--')

    paths = [
        (r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\ARF dil1 10nM_44x66y_free.tif", 0, 474),
        (r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\ARF dil1 10nM_65x38y_free.tif", 0, None),
        (r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\ARF dil1 10nM_68x101y_free.tif", 0, 383),
        (r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\ARF dil1 10nM_98x61y_free.tif", 0, 445),
        (r"C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\ARF dil1 10nM_124x82y_free.tif", 0, None),
    ]

    all_lengths = []
    for path, ti, tf in paths:
        lengths = measure_length_free(path)
        lengths *= MPP

        tf = tf or len(lengths)
        lengths[:ti] = np.nan
        lengths[tf:] = np.nan

        all_lengths.append(lengths.copy())

        # tf = tf or len(lengths)
        # all_lengths.append(lengths.copy())
        # lengths = lengths[ti:tf]
        plt.plot(np.arange(lengths.shape[0]) / FPS, lengths * 1e6, 'grey')
        
    all_lengths = np.array(all_lengths)


    ti = 10
    tf = int(30 * FPS)

    lengths_avg = np.nanmean(all_lengths, axis=0)

    avg_length = np.average(lengths_avg[ti:tf]) * 1e6
    print(f"{avg_length = :.2f}")

    F = wlc_force(avg_length * 1e-6, L=16.5e-6, Lp=50e-9)
    print(f"{F*1e12 = :.2f} pN")





    plt.plot(
        np.arange(ti, len(lengths_avg))/FPS, 
        lengths_avg[ti:] * 1e6,
        label='Free'
        )
    plt.axhline(avg_length, color='C1', linestyle='--')

    
    plt.ylabel('Length ($\mu m$)')
    plt.xlabel('Time ($s$)')
    # plt.axhline(16.5, color='k', linestyle='--')
    plt.xlim([0, 59.9])
    plt.ylim([0, None])
    plt.legend()
    plt.savefig("length_bead.pdf")
    plt.show()