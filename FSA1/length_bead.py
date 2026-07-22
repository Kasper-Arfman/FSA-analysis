import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.constants import k
from pyjacket import filetools, plottools
from tools.length_measurement import measure_length_bead, measure_length_free
plottools.style.use('sprakel_thesis')

DATA_ROOT = r'C:\Users\arfma005\Documents\GitHub\FSA-analysis\data\FSA1'
PATHS_BEAD = [
    (fr"{DATA_ROOT}\ARF dil1 10nM_14x58y_bead.tif", 10, None),
    (fr"{DATA_ROOT}\ARF dil1 10nM_85x88y_bead.tif", 0, None),
    (fr"{DATA_ROOT}\ARF dil1 10nM_146x75y_bead.tif", 0, 463),
]

PATHS_FREE = [
    (fr"{DATA_ROOT}\ARF dil1 10nM_44x66y_free.tif", 0, 474),
    (fr"{DATA_ROOT}\ARF dil1 10nM_65x38y_free.tif", 0, None),
    (fr"{DATA_ROOT}\ARF dil1 10nM_68x101y_free.tif", 0, 383),
    (fr"{DATA_ROOT}\ARF dil1 10nM_98x61y_free.tif", 0, 445),
    (fr"{DATA_ROOT}\ARF dil1 10nM_124x82y_free.tif", 0, None),
]

BEAD_RADIUS = 810e-9 / 2
MPP = 6.5e-6 / 100
FPS = 10
TI, TF = 10, int(30 * FPS)

def wlc_force(r, L, Lp, T=298):
    x = r / L
    if x > 1:  raise ValueError('OVERSTRETCHED POLYMER')
    return k*T/(4*Lp) * ((1-x)**-2 + 4*x - 1 - 3.2*x**2.15)   # 1% error

def measure_lengths_bead(paths: list[str]):
    all_lengths = []
    for path, ti, tf in paths:
        lengths = measure_length_bead(path)
        lengths *= MPP
        lengths -= BEAD_RADIUS

        tf = tf or len(lengths)
        lengths[:ti] = np.nan
        lengths[tf:] = np.nan

        all_lengths.append(lengths.copy())
    return np.array(all_lengths)

def measure_lengths_free(paths: list[str]):
    all_lengths = []
    for path, ti, tf in paths:
        lengths = measure_length_free(path)
        lengths *= MPP

        tf = tf or len(lengths)
        lengths[:ti] = np.nan
        lengths[tf:] = np.nan

        all_lengths.append(lengths.copy())
    return np.array(all_lengths)

# Bead
lengths_bead = measure_lengths_bead(PATHS_BEAD)
lengths_bead_avg = np.nanmean(lengths_bead, axis=0)
average_length_bead = np.average(lengths_bead_avg[TI:TF])
time_bead = np.arange(len(lengths_bead_avg)) / FPS

# Free
lengths_free = measure_lengths_free(PATHS_FREE)
lengths_free_avg = np.nanmean(lengths_free, axis=0)
average_length_free = np.average(lengths_free_avg[TI:TF])
time_free = np.arange(len(lengths_free_avg)) / FPS

F_bead = wlc_force(average_length_bead, L=16.5e-6, Lp=50e-9)
F_free = wlc_force(average_length_free, L=16.5e-6, Lp=50e-9)
print(f"Bead: {F_bead*1e12:.2f} pN\nFree: {F_free*1e12:.2f} pN")

if True:
    plt.figure(figsize=(5, 3))
    plt.subplots_adjust(left=0.55/5, right=0.99, top=0.99, bottom=0.45/3)

    # Bead traces
    for length_trace in lengths_bead:
        plt.plot(time_bead[TI:], length_trace[TI:] * 1e6, 'grey')
    plt.plot(time_bead[TI:], lengths_bead_avg[TI:] * 1e6)
    plt.axhline(average_length_bead*1e6, color='C0', linestyle='--')

    # Free traces
    for length_trace in lengths_free:
        plt.plot(time_free[TI:], length_trace[TI:] * 1e6, 'grey')
    plt.plot(time_free[TI:], lengths_free_avg[TI:] * 1e6)
    plt.axhline(average_length_free*1e6, color='C1', linestyle='--')

    plt.ylabel('Length ($\mu m$)')
    plt.xlabel('Time ($s$)')
    plt.xlim([0, 59.9])
    plt.ylim([0, None])
    plt.legend()
    plt.savefig("out/length_bead.pdf")
    plt.show()