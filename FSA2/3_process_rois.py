import cv2
import numpy as np
from operator import sub
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
from pyjacket import filetools
from pyjacket.graphtools.skeleton import critical_points
from pyjacket.graphtools.shortest_path import max_shortest_path
from pyjacket.graphtools.grid import graph_from_grid
from _script_params import *
from tools.dna_filter import isolate_largest
from tools.iter_config import iter_measurements

LOG_FILE = 'errors.txt'
OVERRIDE = True
DX = ROI_MARGIN[0]
fm = filetools.FileManager(ANALYSIS_MANUAL, ANALYSIS_ROOT)

def main():
    # Wipe the log file
    with open(LOG_FILE, 'w') as f:
        f.write('')

    for measurement_path, kw_meas in iter_measurements(CONFIG):
        if TARGET:
            if TARGET.lower() not in measurement_path.lower():  continue
            
        print(f"\n\n=== {measurement_path} ===")
        process_measurement(fm, measurement_path, **kw_meas)

def process_measurement(fm: filetools.FileManager, measurement_path: str, **kw_meas):
    fm.rel_path = measurement_path

    # Build kwargs
    try:
        kw_meas = {**fm.read_json('config.json'), **kw_meas}
    except FileNotFoundError:
        pass
    if 'dna_type' not in kw_meas:
        pattern = measurement_path.lower()
        if '_lam' in pattern:  DNA_TYPE = 'lambda'
        elif '_dd3' in pattern:   DNA_TYPE = 'dd3'
        elif '_dd7' in pattern:   DNA_TYPE = 'dd7'
        else:  raise ValueError(f'Cannot interpret DNA type from:\n{measurement_path = }')
        kw_meas['dna_type'] = DNA_TYPE
    

    # Check continue
    if not fm.exists('', folder='rois_tif', dst=True):
        with open(LOG_FILE, 'a+')as f:
            f.write(f'{fm.rel_path}: missing rois_tif output folder\n')
        print('WARNING: missing rois_tif folder. Do process_raw.py first!')
        return

    # Check 
    for roi_name in fm.iter_dir('rois_tif', '.tif', dst=True):
        print(f"  > {roi_name}")

        if not OVERRIDE \
            and fm.exists(roi_name, 'rois_tif', dst=True) \
            and fm.exists(roi_name, 'rois_tif_filt', dst=True) \
            and fm.exists(roi_name, 'rois_tif_filt_blur', dst=True) \
            and fm.exists(roi_name, 'rois_tif_bin', dst=True):
            print('skipped  (exists)')
            continue

        try:
            process_ROI(fm, roi_name, **kw_meas)
        except Exception as e:
            print(e)
            with open(LOG_FILE, 'a+')as f:
                f.write(f'{fm.rel_path}\{roi_name}: {e}\n')

            raise e

    return

def process_ROI(fm: filetools.FileManager, roi_name, **kwargs):
    DNA_CH = kwargs.get('dna_ch', DNA_CH_DEFAULT)
    PROT_CH = 1 - DNA_CH
                
    # - read the roi data
    roi = fm.read_img(roi_name, folder='rois_tif', dst=True)
    
    if fm.src_root != fm.dst_root:
        fm.write_img(roi_name, roi, folder='rois_tif', color='mg')

    protein_channel = roi[..., PROT_CH]
    dna_channel = roi[..., DNA_CH]

    # - 3D Gaussian blur
    if True:
        dna_blurred = gaussian_filter(dna_channel, sigma=DNA_BLUR)
        protein_blurred = gaussian_filter(protein_channel, sigma=PROT_BLUR)
        if True:    
            tmp = np.stack([protein_blurred, dna_blurred], axis=-1)
            fm.write_img(roi_name, tmp, folder='1_preproc', color='mg')

    # - Thresholding
    if True:
        # > Set threshold levels
        dna_threshold = sum([
            (1-DNA_STRICTNESS)*np.percentile(dna_blurred[DNA_BLUR[0]], NOISE_PERCENTILE), 
            (  DNA_STRICTNESS)*np.percentile(dna_blurred, SIGNAL_PERCENTILE)])
        prot_threshold = sum([
            np.percentile(protein_blurred, 50),  # <noise mean>
            3 * sub(*np.percentile(protein_blurred, [84, 16]))//2])
        
        #   > Binarize
        i_dna = np.zeros_like(dna_blurred)
        i_dna[dna_blurred > dna_threshold] = 255
        i_prot = np.zeros_like(protein_blurred)
        i_prot[protein_blurred  > prot_threshold] = 255

        #   > Erode / Dilate to separate or connect features
        i_dna = cv2.erode(i_dna, EROSION1, iterations=1)
        i_dna = cv2.dilate(i_dna, DILATION1, iterations=1)

        #   > isolate only the maximal contour
        i_dna = isolate_largest(i_dna)
        if True:
            tmp = np.stack([i_prot, i_dna], axis=-1)
            fm.write_img(roi_name, tmp, folder='2_bin', color='mg')

    # - Longest chain in skeleton
    if True:
        # > Skeleton
        skeleton = np.zeros_like(i_dna)
        for i, frame in enumerate(i_dna):
            skeleton[i] = skeletonize(frame).astype(np.uint8)

        # > Branchless chain
        chain = np.zeros_like(skeleton)
        for i, frame in enumerate(skeleton):
            endpoints, branch_points = critical_points(frame)
            best_pair, path, max_length = max_shortest_path(graph_from_grid(frame), endpoints)
            for y, conc in path:
                chain[i, y, conc] = 1
        if True:
            tmp = np.stack([i_prot, 255*chain], axis=-1)
            fm.write_img(roi_name, tmp, folder='3_chain', color='mg')

    # - DNA filter
    if True:
        # > Obtain mask
        i_dna = gaussian_filter(chain.astype(np.float32), sigma=CHAIN_BLUR)
        ma = np.percentile(i_dna, 98)
        i_dna = 255 * i_dna / ma
        i_dna[i_dna <= 255 - THICKNESS] = 0
        i_dna[i_dna  > 255 - THICKNESS] = 255
        i_dna = i_dna.astype(np.uint8)

        i_prot = np.bool_(i_dna) * i_prot
        if True:
            tmp = np.stack([i_prot, i_dna], axis=-1)
            fm.write_img(roi_name, tmp, folder='rois_tif_bin', color='mg')

        # > Apply mask
        if True:
            tmp = np.stack([
                np.bool_(i_dna) * protein_channel, 
                np.bool_(i_dna) * dna_channel
                ], axis=-1)
            fm.write_img(roi_name, tmp, folder='rois_tif_filt', color='mg')

        # > Blur -> Apply mask
        if True:
            tmp = np.stack([
                np.bool_(i_dna) * gaussian_filter(protein_channel, sigma=MINOR_BLUR), 
                np.bool_(i_dna) * gaussian_filter(dna_channel, sigma=MINOR_BLUR),
                ], axis=-1)
            fm.write_img(roi_name, tmp, folder='rois_tif_filt_blur', color='mg')

    return

if __name__ == '__main__':
    main()
    print('\nFinished!')