import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from pyjacket import filetools, plottools
from tools.iter_config import iter_bundle, iter_dataset
from _script_params import *
import numpy as np
plottools.style.use('sprakel_thesis')

DATASET_NAME = 'Lambda_FL_concentration'
fm = filetools.FileManager(ANALYSIS_ROOT, ANALYSIS_ROOT)

def main():
    # Data collection
    coverages = defaultdict(list)
    for bundle_name in iter_dataset(DATASET_NAME, CONFIG):
        print(f"\n\n=== {bundle_name} ===")
        c = infer_concentration(bundle_name)
        for meas_path, kwargs in iter_bundle(bundle_name, CONFIG):
            coverages[c] += measure_coverage(meas_path, **kwargs)

    # Data averaging
    conc = list(sorted(coverages))
    print(f"{conc = }")
    cov = np.array([np.average(coverages[c]) for c in conc])
    print(f'{cov = }')

    cov_sem = np.array([stats.sem(coverages[c]) for c in conc])

    # Baseline correction
    if True:
        cov = ((cov - cov[0]) / (1 - cov[0]))

    # Plotting
    plt.figure()
    plottools.shaded_plot(conc, cov, cov_sem)
    plt.ylabel('Coverage [-]')
    plt.xlabel('Concentration [$nM$]')
    fm.write_pickle(f'coverage_{DATASET_NAME}', [conc, cov, cov_sem])
    plt.show()

def infer_concentration(bundle_name: str):
    m = re.search(r'_(\d+)nM', bundle_name)
    conc = int(m.group(1))
    return conc

def measure_coverage(rel_path, **kwargs):
    fm.rel_path = rel_path

    cov = []
    for roi_name in fm.iter_dir('rois_tif', '.tif'):
        print(rel_path, roi_name)
        if roi_name in (kwargs.get('exclude') or []):
            print(f' > excluded')
            continue

        ti = kwargs.get('ti', TI_DEFAULT)
        tf = kwargs.get('tf', TF_DEFAULT)

        i_roi = fm.read_img(roi_name, folder='rois_tif_bin')
        DNA_area = np.sum(np.bool_(i_roi[ti:tf, ..., DNA_CH_DEFAULT]), axis=(1, 2)) + 1e-8
        prot_area = np.sum(np.bool_(i_roi[ti:tf, ..., PROT_CH_DEFAULT]), axis=(1, 2))
        coverage = np.median(prot_area/DNA_area)
        cov.append(coverage)

    fm.rel_path = ''
    return cov

if __name__ == "__main__":
    main()
    print('\nFinished Successfully')