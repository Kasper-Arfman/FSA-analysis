import pandas as pd
import pyjacket as pj
import numpy as np
from pprint import pprint
from pyjacket import arrtools, filetools
from tools.rotate import rotate_frame, rotate_points
from tools.intensity import intensity_histogram, percentile
from tools.iter_config import iter_measurements
from _script_params import *

TEST = True  # <==== Disable to analyse your full dataset !!!
AUTO_SCALE = False
OVERRIDE = True

def main():
    fm        = filetools.FileManager(DATA_ROOT, ANALYSIS_ROOT)
    fm_manual = filetools.FileManager(ANALYSIS_MANUAL, ANALYSIS_MANUAL)
    for rel_path, kwargs in iter_measurements(CONFIG):
        if TARGET is not None:
            if TARGET.lower() not in rel_path.lower():  continue

        elif not OVERRIDE:
            if fm.exists('rois_tif', rel_path, dst=True):
                continue

        try:
            process_alex(fm, fm_manual, rel_path) 
        except Exception as e:
            print(e)
            raise e

def preprocess_image_stack(angle: float=0, kernel_size=101):
    """Data manipulation for processed data visualization"""

    def inner(image_handle: filetools.ImageHandle, i: int) -> np.ndarray:
        if i % 10 == 0:  print(f'- frame {i:4d}')
        frame = image_handle.get(i)
        dtype_out = np.uint8
        tmp = np.empty(frame.shape, dtype=dtype_out)
        for ch in range(frame.shape[-1]):
            f = frame[..., ch]
            f = arrtools.gaussian_blur(f, (1, 1))
            f = arrtools.distribute(f, 0.05, 0.05, dtype=dtype_out)

            f = arrtools.subtract_median(f, kernel_size)
            # f = arrtools.subtract_uint(f, arrtools.gaussian_blur(f, (21, 21)))  # Faster but slightly worse alternative
            tmp[..., ch] = f
        frame = tmp

        frame = rotate_frame(frame, -angle)
        return frame
    
    return inner
        
def process_alex(fm: filetools.FileManager, fm_manual: filetools.FileManager, rel_path_name: str):
    print(f'\n\n\n== Processing {rel_path_name}')
    if '_Lam_' in rel_path_name:
        roi_length = ROI_LENGTH['lambda']
    elif '_dd' in rel_path_name:
        roi_length = ROI_LENGTH['dashdot']
    else:
        raise ValueError('Unknown DNA type (dashdot/lambda)')
    print(f"{roi_length = :.2f} px")


    print('\n-- Read config')
    if True:
        try:
            conf = fm_manual.read_json('config.json', rel_path_name)
        except FileNotFoundError:
            print('WARNING: no config.json found.')
            conf = dict()
        pprint(conf)

    print(f'\n-- Read Image data (tested for MMStacks only)\n{rel_path_name}')
    if True:
        img_handle = fm.read_img(rel_path_name, lazy=True, unzip=2)

        if TEST:
            img_handle = img_handle[100:200]

        print(f"shape {img_handle.shape} / {img_handle.max_shape}")

    print('\n-- Read ROI positions')
    if True:
        # Will throw error if no positions.csv is found, because then what's the point.
        positions: pd.DataFrame
        positions = fm_manual.read_csv(f"positions.csv", folder=rel_path_name)
        print(f"{positions.shape = }")

    print(f'\n-- Rotate positions')
    if True:
        angle = conf.get('angle', 0)
        positions = pj.df_apply(positions, rotate_points, degrees=-angle, origin=(1024, 1024))
        print(f"{positions.shape = }")

    # Make mp4 of raw data
    if False:
        print('\n== Intensity histogram (May take a while)')
        raw_intensity_histogram = lazy.intensity_histogram(img_handle, color=True)    # (65536, 2)
        plot_intensity_histogram(fm, raw_intensity_histogram, 'hist_raw.png')
        
        print('\n== Brightness scaling')
        scale = lazy.percentile(raw_intensity_histogram, (0.5, 0.995), color=True)
        print(scale)

        print('\n== Export raw data to mp4 (May take a while)')
        img_handle.operator = operate_raw(angle=angle)
        fm.write_img(f"raw.mp4", img_handle, frame_time=FRAME_TIME, scale=scale)

    print('\n-- Brightness scaling')
    if True:
        scale = conf.get('scale', None)
        if scale is None:
            if AUTO_SCALE:
                print('Autoscaling... (may take a while)')
                img_handle.operator = preprocess_image_stack(angle=0)

                # - Computing intensity histogram
                raw_intensity_histogram = intensity_histogram(img_handle, color=True)    # (65536, 2)
                scale = percentile(raw_intensity_histogram, (0.5, 0.995), color=True)

                # - Save this output to a cache file
                conf['scale'] = scale.tolist()
                fm_manual.write_json('config.json', conf, rel_path_name)
            else:  # Manual scaling
                print('Using default.')
                scale = DEFAULT_SCALE
        else:
            print('Using config.json')
        scale = np.array(scale)
        print(scale)

    print('\n-- Processing data ...')
    if True:
        img_handle.operator = preprocess_image_stack(angle=angle)
        fm.write_img('proc.tif', img_handle)
        img_handle = fm.read_img('proc.tif', dst=True)

    # Make mp4 of processed data
    if False:
        print('\n== Export proc data to mp4 (May take a while)')
        fm.write_img(f"proc.mp4", img_handle, folder=rel_path_name, frame_time=FRAME_TIME, scale=scale)

    print('\n-- ROI analysis')
    fm.rel_path = rel_path_name
    DX, DY = ROI_MARGIN
    for i, pos in enumerate(positions.itertuples()):
        print(f'\n  > ROI {i}')
        x, y = int(pos.x), int(pos.y)

        roi = np.array([frame[y-DY:y+DY, x-roi_length-DX: x+DX] for frame in img_handle])

        # Guarantee protein ch, DNA-channel
        dna_ch = conf.get('dna_ch', DNA_CH_DEFAULT)
        roi = np.stack([roi[..., 1-dna_ch], roi[..., dna_ch]], axis=-1)

        fm.write_img(f"roi_{i}.tif", roi, folder='rois_tif', color='mg')
    fm.rel_path = ''
    fm.delete('proc.tif')

if __name__ == '__main__':
    main()
    print('\nFinished successfully!')