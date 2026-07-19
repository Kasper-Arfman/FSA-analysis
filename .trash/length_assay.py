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
from pyjacket import filetools

NEIGHBOR_OFFSETS = [(-1, -1), (-1,  0), (-1,  1), 
                    ( 0, -1),           ( 0,  1), 
                    ( 1, -1), ( 1,  0), ( 1,  1)]

def graph_from_grid(skeleton: np.ndarray):
    """Convert a binary image into a graph
    
    """
    assert skeleton.dtype == np.uint8, ValueError('Skeleton must be np.uint8')
    pixel_positions = set(zip(*np.where(skeleton > 0)))
    
    graph = nx.Graph()
    for pos in pixel_positions:
        for neighbor in neighbor_positions(*pos, filter=pixel_positions):
            graph.add_edge(pos, neighbor, weight=distance.euclidean(pos, neighbor))
    return graph

def neighbor_positions(r, c, filter=None):
    """Find all neighboring coordinates.
    Optionally, return only coordinates that are present in the filter
    """
    for dr, dc in NEIGHBOR_OFFSETS:
        neighbor = (r + dr, c + dc)
        if filter is not None and neighbor not in filter:
            continue
        yield neighbor


def max_shortest_path(graph: nx.Graph, endpoints: list[tuple]):
    """Find the pair of endpoints whose shortest path is maximal and return it as a binary image."""
    endpoints = [tuple(yx) for yx in endpoints]
    
    pair, path, path_length = (None, [], -float('inf'))
    for i, p1 in enumerate(endpoints):
        for p2 in endpoints[i+1:]:
            try:
                pair = (p1, p2)
                pth = nx.shortest_path(graph, source=p1, target=p2, weight='weight')
                length = sum(distance.euclidean(pth[k], pth[k+1]) for k in range(len(pth)-1))
                if length > path_length:
                    pair, path, path_length = (pair, pth, length)
            except nx.NetworkXNoPath:
                continue
    return pair, path, path_length

BRANCH_PATTERNS = np.array([
    [[0, 1, 0],
     [1, 1, 1],
     [0, 0, 0]],

    [[1, 0, 1,],
     [0, 1, 0,],
     [1, 0, 0,]],

    [[1, 0, 1],
     [0, 1, 0],
     [0, 1, 0]],

    [[0, 1, 0],
     [1, 1, 0],
     [0, 0, 1]],

    [[0, 0, 1],
     [1, 1, 1],
     [0, 1, 0]],

    [[1, 0, 0],
     [1, 1, 1],
     [0, 1, 0]],

    [[0, 1, 0],
     [1, 1, 0],
     [0, 1, 0]],

    [[0, 0, 0],
     [1, 1, 1],
     [0, 1, 0]],

    [[0, 1, 0],
     [0, 1, 1],
     [0, 1, 0]],

    [[1, 0, 0],
     [0, 1, 0],
     [1, 0, 1]],

    [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 1]],

    [[1, 0, 1],
     [0, 1, 0],
     [0, 0, 1]],

    [[1, 0, 0],
     [0, 1, 1],
     [1, 0, 0]],

    [[0, 1, 0],
     [0, 1, 0],
     [1, 0, 1]],

    [[0, 0, 1],
     [1, 1, 0],
     [0, 0, 1]],

    [[0, 0, 1],
     [1, 1, 0],
     [0, 1, 0]],

    [[1, 0, 0],
     [0, 1, 1],
     [0, 1, 0]],

    [[0, 1, 0],
     [0, 1, 1],
     [1, 0, 0]],

    [[1, 1, 0],
     [0, 1, 1],
     [0, 1, 0]],

    [[0, 1, 0],
     [1, 1, 1],
     [1, 0, 0]],

    [[0, 1, 0],
     [1, 1, 0],
     [0, 1, 1]],

    [[0, 1, 0],
     [0, 1, 1],
     [1, 1, 0]],

    [[0, 1, 0],
     [1, 1, 1],
     [0, 0, 1]],

    [[0, 1, 1],
     [1, 1, 0],
     [0, 1, 0]],

    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],

    [[1, 0, 1],
     [0, 1, 0],
     [1, 0, 1]],
    
    ])

NEIGHBOR_KERNEL = np.array([[1, 1, 1], 
                            [1, 0, 1], 
                            [1, 1, 1]])

def is_branch_point(skeleton: np.ndarray, y, x):

    padded_region = np.zeros((3, 3), dtype=skeleton.dtype)


    # Get the valid region indices
    y_start, y_end = max(0, y-1), min(skeleton.shape[0], y+2)
    x_start, x_end = max(0, x-1), min(skeleton.shape[1], x+2)

    # Compute the corresponding indices in the 3x3 patch
    patch_y_start, patch_y_end = 1 - (y - y_start), 1 + (y_end - y)
    patch_x_start, patch_x_end = 1 - (x - x_start), 1 + (x_end - x)

    # Copy valid data from skeleton into the 3x3 region
    padded_region[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = skeleton[y_start:y_end, x_start:x_end]


    return any((
        np.all(pattern==padded_region) \
            for pattern in BRANCH_PATTERNS
    ))



def critical_points(skeleton: np.ndarray):
    """Finds end points and branch points for a skeleton image"""
    assert skeleton.dtype == np.uint8, ValueError('Skeleton must be np.uint8')
    
    neighbor_count = convolve(skeleton, NEIGHBOR_KERNEL, mode="constant", cval=0)

    end_points = np.argwhere(skeleton & (neighbor_count == 1))

    branch_candidates = np.argwhere(skeleton & (neighbor_count >= 3))
    branch_points = [yx for yx in branch_candidates if is_branch_point(skeleton, *yx)]
    return end_points, branch_points

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
DNA_BLUR = (3, 1.5, 1.5)  # (1, 1, 1) 
DNA_STRICTNESS = 0.45  #0.40
NOISE_PERCENTILE = 50
SIGNAL_PERCENTILE = 92
EROSION1 = np.ones((2, 2), np.uint8)  # y, x
DILATION1 = np.ones((2, 2), np.uint8)  # y, x

# ARF sample with beads and some without
# ARF is going to compact the DNA
# Measure how quickly this happens

# DNA length over time

path = r'data\ARF dil1 10nM_14x58y_bead.tif'

img = tifffile.imread(path)

# 3D gaussian blur
dna_blurred = gaussian_filter(img, sigma=DNA_BLUR)

# > Set threshold levels
dna_threshold = sum([
    (1-DNA_STRICTNESS)*np.percentile(dna_blurred[DNA_BLUR[0]], NOISE_PERCENTILE), 
    (  DNA_STRICTNESS)*np.percentile(dna_blurred, SIGNAL_PERCENTILE)])

# > Binarize
i_dna = np.zeros_like(dna_blurred)
i_dna[dna_blurred > dna_threshold] = 255


# > Erode / Dilate to separate or connect features
# i_dna = cv2.erode(i_dna, EROSION1, iterations=1)
# i_dna = cv2.dilate(i_dna, DILATION1, iterations=1)
i_dna = np.astype(i_dna, np.uint8)

# > isolate only the maximal contour
i_dna = isolate_largest(i_dna)

filetools.write_img('test.tif', i_dna)


# > Extract lengths
lengths = []
for i, frame0 in enumerate(i_dna):
    line = frame0.sum(axis=0)  # number of white pixels along x
    pixel_indices = np.argwhere(line > 0)

    bead_peak = np.argmax(line)



    lengths.append(bead_peak - pixel_indices[0])
lengths = np.array(lengths)


lengths_smooth = median_filter(lengths, size=30)


plt.plot(lengths)
plt.plot(lengths_smooth)
plt.show()


# for _ in range(10):
#     i_dna = cv2.erode(i_dna, np.ones((2, 2)), iterations=1)
#     i_dna = cv2.dilate(i_dna, np.ones((2, 2)), iterations=1)

# # > Longest chain in skeleton
# skeleton = np.zeros_like(i_dna)
# for i, frame in enumerate(i_dna):
#     skeleton[i] = skeletonize(frame).astype(np.uint8)

# # > Branchless chain
# chain = np.zeros_like(skeleton)
# for i, frame in enumerate(skeleton):
#     endpoints, branch_points = critical_points(frame)
#     best_pair, path, max_length = max_shortest_path(graph_from_grid(frame), endpoints)
#     for y, conc in path:
#         chain[i, y, conc] = 1


# Find the minimal xvalue and th

# # - DNA filter
# if True:
#     # > Obtain mask
#     i_dna = gaussian_filter(chain.astype(np.float32), sigma=CHAIN_BLUR)
#     ma = np.percentile(i_dna, 98)
#     i_dna = 255 * i_dna / ma
#     i_dna[i_dna <= 255 - THICKNESS] = 0
#     i_dna[i_dna  > 255 - THICKNESS] = 255
#     i_dna = i_dna.astype(np.uint8)

#     i_prot = np.bool(i_dna) * i_prot
#     if True:
#         tmp = np.stack([i_prot, i_dna], axis=-1)
#         fm.write_img(roi_name, tmp, folder='rois_tif_bin', color='mg')

#     # > Apply mask
#     if True:
#         tmp = np.stack([
#             np.bool(i_dna) * protein_channel, 
#             np.bool(i_dna) * dna_channel
#             ], axis=-1)
#         fm.write_img(roi_name, tmp, folder='rois_tif_filt', color='mg')

#     # > Blur -> Apply mask
#     if True:
#         tmp = np.stack([
#             np.bool(i_dna) * gaussian_filter(protein_channel, sigma=MINOR_BLUR), 
#             np.bool(i_dna) * gaussian_filter(dna_channel, sigma=MINOR_BLUR),
#             ], axis=-1)
#         fm.write_img(roi_name, tmp, folder='rois_tif_filt_blur', color='mg')
# longest chain




img = i_dna
# img = chain

# print(img)


plt.imshow(img[T], cmap='Greys_r')
plt.show()