import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tools.animate_dist import animate_distribution
from pyjacket import arrtools, filetools, plottools
from pyjacket.filetools import FileManager
from tools.iter_config import iter_measurements
from _script_params import *
plottools.style.use('sprakel_thesis')

import warnings
warnings.filterwarnings("ignore", message=".*The figure layout has changed to tight.*")

ROIS_PROC = 'rois_tif_filt_blur'
LENGTH_THRESHOLD = 0.0
ANIMATE = False
SHOW = False

def main():
    fm = filetools.FileManager(ANALYSIS_ROOT, ANALYSIS_ROOT)
    for measurement_path, kw_meas in iter_measurements(CONFIG):
        if TARGET:
            if TARGET.lower() not in measurement_path.lower():  continue

        print(f"\n\n=== Processing {measurement_path} ===")
        process_measurement(fm, measurement_path, **kw_meas, show=SHOW, animate=ANIMATE)

def process_measurement(fm: FileManager, measurement_path: str, animate=False, show=False, rois_proc=ROIS_PROC, **kw_meas):
    fm.rel_path = measurement_path

    # Build kwargs
    try:
        kw_meas = {**fm.read_json('config.json'), **kw_meas}
    except FileNotFoundError:
        pass

    # Read DNA type
    if '_Lam_' in measurement_path:
        roi_length = ROI_LENGTH['lambda']
    elif '_dd' in measurement_path:
        roi_length = ROI_LENGTH['dashdot']
    else:
        raise ValueError('Cannot read ROI_length')

    # Obtain profiles
    roi_names = []
    profiles = []
    lengths = []
    for roi_name in fm.iter_dir(rois_proc, '.tif'):
        # print(f"  > {roi_name}")

        if roi_name in kw_meas.get('exclude', set()):
            print(f'  > {roi_name } excluded by user')
            continue

        ti = kw_meas.get('ti', TI_DEFAULT)
        tf = kw_meas.get('tf', TF_DEFAULT)
        roi = fm.read_img(roi_name, rois_proc)[ti:tf]

        profile, n, median_length = process_ROI(roi, **kw_meas)
        print(f"  > {roi_name} - {median_length:.0f} px")

        if n is None:  
            print('skipping (0 frames)')
            continue
        if np.sum(profile[..., PROT_CH_DEFAULT]) == 0:  
            print('skipping (no binding events)')
            continue

        

        profiles += [(profile, n)]
        roi_names += [roi_name]
        lengths += [median_length]

        if animate:
            ani = animate_distribution(roi)
            fm.write(f'{roi_name}.mp4', ani, fps=30, folder='mp4_profile')
            # exit()


    # Resample profiles
    x_interp = np.linspace(0, 1, num=roi_length + 1)
    y_values = []
    weights = []
    for i, (profile, n) in enumerate(profiles):
        x = np.linspace(0, 1, num=len(profile))
        y = profile[..., PROT_CH_DEFAULT]
        y_interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)(x_interp)
        y_values += [y_interp]
        # plt.plot(x_interp, y_interp)

        weights.append(n)
    weights = np.array(weights)
    y_values = np.array(y_values)

    # Profiles
    df = pd.DataFrame(y_values.T, columns=roi_names)
    fm.write_csv('profiles_interp.csv', df)

    # Lengths
    df = pd.DataFrame(lengths, roi_names)
    fm.write_csv('lengths.csv', df)
    
    # Plot resampled profiles

    # Pool data via weighted average
    for y in y_values:
        plt.plot(x_interp, y)
    average = np.average(y_values, axis=0, weights=weights)
    plt.plot(x_interp, average, 'k--')
    fm.savefig('profile.pdf', close=not show)
    if show:
        plt.show()

    return y_values, n

def process_ROI(roi: np.ndarray, ti=None, tf=None, length_threshold=LENGTH_THRESHOLD, **kwargs):
    Nframe = roi.shape[0]
    if Nframe == 0:
        # print(f"WARNING: empty roi data")
        return np.array([]), None

    lengths = dna_lengths(roi)
    median_length = np.median(lengths) 
    theoretical_length = ROI_LENGTH['dashdot']

    # print(f'length = {median_length:.0f} / {theoretical_length} px', end=' ')

    if median_length <= length_threshold * theoretical_length:
        # print('(excluded)')
        return np.array([]), None, -1
    else:
        # print('')
        pass


    # kymograph
    DX = ROI_MARGIN[0]
    kym = np.sum(roi, axis=1)
    intensity_profile = np.average(kym, axis=0)[-int(round(median_length))-DX:-DX]
    return intensity_profile, Nframe, median_length

def dna_lengths(roi: np.ndarray):
    lengths = np.zeros(roi.shape[0])
    for i, frame in enumerate(roi[..., DNA_CH_DEFAULT]):
        idx, = np.nonzero(np.sum(frame, axis=0))
        lengths[i] = (idx[-1] - idx[0]) if idx.size != 0 else -1
    return lengths

def measurement_kwargs(fm: FileManager, measurement_path, yaml_file: dict):
    kwargs: dict = yaml_file['measurements']
    for key in fm.explode(measurement_path):
        kwargs = kwargs.get(key) or {}
    return kwargs


if __name__ == '__main__':
    main()
    print('\nFinished!')