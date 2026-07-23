import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import k as KB
from scipy.ndimage import gaussian_filter1d
np.set_printoptions(precision=2)

KT = KB * 298
MIN_FREQ = 1  # Prevent inf energy
NAME, FILE_PATH = [
    ('lambda',  r'FSA2\tools\seq\lambda.fasta'),
    ('ASkb12',  r'FSA2\tools\seq\ASkb12.fasta'),
    ('IR7kb12', r'FSA2\tools\seq\IR7kb12.fasta'),
][2]


def read_fasta(file_path, encode=True):
    r = ''
    with open(file_path) as f:
        dna = f.readlines()
        for line in dna:
            if line.startswith('>'):
                continue
            r += line.strip('\n')

    if encode:
        MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        r = np.array([MAP[b] for b in r.upper()], dtype=np.uint8)
    return r

def sliding_window(arr, n):
    for i in range(len(arr)-n):
        yield arr[i: i+n]

def PFM_to_PWM(pfm: np.ndarray, min_freq):
    """Computes the energy matrix from the frequency matrix"""
    pfm[pfm<min_freq] = min_freq
    ppm = pfm / pfm.sum(axis=0)
    consensus = ppm[np.argmax(ppm, axis=0), np.arange(pfm.shape[-1])]
    pwm = -KT * np.log(ppm / consensus)
    return pwm

def binding_probability(template, PWM):
    L_template = len(template)
    L_motif = PFM.shape[-1]
    I = np.arange(L_motif)

    energy = np.zeros(L_template - L_motif)
    for i, motif in enumerate(sliding_window(template, L_motif)):
        energy[i] = np.sum(PWM[motif, I])
    return np.exp(-energy/KT)


template = read_fasta(FILE_PATH, encode=True)
PFM = np.array([  # AtARF1
        [  8,   2,   7,   1,   2,   4],  # 0: A
        [ 88,   2,   1, 992,   2,   1],  # 1: C
        [  4, 993,   3,   2, 993, 993],  # 2: G 
        [901,   2, 989,   4,   4,   3],  # 3: T
    ], dtype=np.float32)
PWM = PFM_to_PWM(PFM, min_freq=MIN_FREQ)
print(f"PMW:\n{PWM}")

p_F = binding_probability(template, PWM)
p_R = binding_probability(template, PWM[::-1, ::-1])
p = (p_F + p_R) / 2

df = pd.DataFrame()
df['E'] = -KT * np.log(p)
df['p'] = p
df.to_csv(f'E_{NAME}.csv')

sigma = 500  # bases
p = gaussian_filter1d(p, sigma, mode='constant')

if True:
    plt.plot(p)
    plt.show()