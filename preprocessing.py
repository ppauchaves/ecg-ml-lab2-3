import os
import numpy as np
import pandas as pd
import wfdb
import ast
from tqdm import tqdm
from scipy import signal
from ecgdetectors import Detectors
import pickle

# === PATHS & PARAMETERS ===
DATA_PATH = "/Users/pauchaves/Documents/Mathematical Engineering in Data Science/3rd Year/3rd Trimester/COMPBIOMED/physionet.org/files/ptb-xl/1.0.3"
SAVE_PATH = "/Users/pauchaves/Documents/Mathematical Engineering in Data Science/3rd Year/3rd Trimester/COMPBIOMED/ecg-ml-lab2-3"
SAMPLING_RATE = 100
GROUP5_CLASSES = {'NORM', 'STTC', 'HYP'}

os.makedirs(SAVE_PATH, exist_ok=True)

# === 1. LOAD METADATA ===
print("[INFO] Loading metadata...")
Y = pd.read_csv(os.path.join(DATA_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

agg_df = pd.read_csv(os.path.join(DATA_PATH, 'scp_statements.csv'), index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# === 2. FILTER FOR EXCLUSIVE GROUP 5 CLASSES ONLY ===
print("[INFO] Filtering for exclusively Group 5 classes (NORM, STTC, HYP)...")
def is_exclusively_group5(superclass_list):
    return set(superclass_list).issubset(GROUP5_CLASSES)

Y = Y[Y['diagnostic_superclass'].apply(is_exclusively_group5)]

# === 3. LOAD RAW ECG SIGNALS ===
def load_raw_data(df, sampling_rate, base_path):
    data = []
    for f in tqdm(df.filename_lr, desc="Loading ECGs"):
        full_path = os.path.join(base_path, f)
        if not os.path.exists(full_path + ".hea"):
            print(f"[WARN] File not found: {full_path}.hea")
            continue
        signal_data, _ = wfdb.rdsamp(full_path)
        data.append(signal_data)
    return np.array(data)

print("[INFO] Loading raw ECG signals (100Hz)...")
X = load_raw_data(Y, SAMPLING_RATE, DATA_PATH)

# === 4. FILTERING ===
print("[INFO] Applying Butterworth filtering and detrending...")
hp_cutoff = 0.5
lp_cutoff = 45.0
b_hp, a_hp = signal.butter(2, hp_cutoff, btype='high', fs=SAMPLING_RATE)
b_lp, a_lp = signal.butter(2, lp_cutoff, btype='low', fs=SAMPLING_RATE)

def filter_and_denoise(ecg):
    ecg_filt = signal.filtfilt(b_hp, a_hp, ecg, axis=0)
    ecg_filt = signal.filtfilt(b_lp, a_lp, ecg_filt, axis=0)
    ecg_filt = signal.detrend(ecg_filt, axis=0)
    return ecg_filt

# === 5. R-PEAK DETECTION ===
print("[INFO] Detecting R-peaks and computing BPM...")
detectors = Detectors(SAMPLING_RATE)

X_filtered = []
R_peaks_all = []
BPM_all = []

for i in tqdm(range(X.shape[0]), desc="Processing signals"):
    ecg_raw = X[i]
    ecg_filt = filter_and_denoise(ecg_raw)
    X_filtered.append(ecg_filt)

    try:
        r_peaks = detectors.pan_tompkins_detector(ecg_raw[:, 0])
        bpm = 60 * len(r_peaks) / 10
    except:
        r_peaks = []
        bpm = np.nan

    R_peaks_all.append(r_peaks)
    BPM_all.append(bpm)

X_filtered = np.array(X_filtered)
Y = Y.iloc[:len(X_filtered)]
Y['bpm'] = BPM_all

# === 6. CLEAN IRRELEVANT COLUMNS ===
print("[INFO] Dropping irrelevant metadata columns...")
columns_to_keep = ['age', 'sex', 'diagnostic_superclass', 'bpm']
Y_clean = Y[columns_to_keep].copy()
Y_clean = Y_clean.dropna(subset=['age', 'sex', 'bpm'])  # Drop rows missing core info
X_filtered = X_filtered[:len(Y_clean)]

# === 7. SAVE OUTPUT ===
print("[INFO] Saving filtered signals, labels, and R-peaks...")
np.save(os.path.join(SAVE_PATH, 'X_filtered_group5.npy'), X_filtered)
Y_clean.to_csv(os.path.join(SAVE_PATH, 'Y_labels_group5.csv'))
with open(os.path.join(SAVE_PATH, 'r_peaks_group5.pkl'), 'wb') as f:
    pickle.dump(R_peaks_all, f)

print("[✅ DONE] Preprocessing complete — filtered, cleaned, labeled dataset ready.")
