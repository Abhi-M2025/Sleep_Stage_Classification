import mne
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_DIR = "./raw/"
TRAIN_DIR = "./processed_train_data/"
TEST_DIR = "./processed_test_data/"
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Map the sleep stage to a numerical output
stage_map = {
    'Sleep stage W': 0,
    'Sleep stage N1': 1,
    'Sleep stage N2': 2,
    'Sleep stage N3': 3,
    'Sleep stage R': 4
}

def process_subject(sn, out_dir):
    # Processing step
    edf_path = os.path.join(DATA_DIR, f"{sn}.edf")
    ann_path = os.path.join(DATA_DIR, f"{sn}_sleepscoring.edf")

    if not os.path.exists(edf_path):
        print(f"Missing {edf_path}, skipping")
        return
    if not os.path.exists(ann_path):
        print(f"Missing {ann_path}, skipping")
        return

    # Load PSG
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    print(f"\n{sn} ORIGINAL CHANNELS:", raw.ch_names)

    # Load scoring annotations
    ann = mne.read_annotations(ann_path)
    raw.set_annotations(ann)

    # Standardize channel/signal naming across different .edf files
    eeg_c4 = [ch for ch in raw.ch_names if "C4-M1" in ch]
    eeg_c3 = [ch for ch in raw.ch_names if "C3-M2" in ch]

    eog_e1 = [ch for ch in raw.ch_names if "E1-M2" in ch]
    eog_e2 = [ch for ch in raw.ch_names if "E2-M2" in ch]

    # EMG may appear with many names
    emg_ch = [ch for ch in raw.ch_names if "emg" in ch.lower()]

    # HARD REQUIREMENT: must have both EEGs, both EOGs, and EMG
    if not (eeg_c4 and eeg_c3 and eog_e1 and eog_e2 and emg_ch):
        print(f"{sn} missing required channels — skipping.")
        return

    # Keep only these channels
    keep = eeg_c4 + eeg_c3 + eog_e1 + eog_e2 + emg_ch
    raw.pick_channels(keep)

    # Rename channels
    rename_map = {
        eeg_c4[0]: "C4-M1",
        eeg_c3[0]: "C3-M2",
        eog_e1[0]: "E1-M2",
        eog_e2[0]: "E2-M2",
        emg_ch[0]: "EMG"
    }
    raw.rename_channels(rename_map)

    # Manually create EOG signal measurement based off E1-M2 and E2-M2 signals
    if "E1-M2" in raw.ch_names and "E2-M2" in raw.ch_names:
        e1 = raw.copy().pick_channels(["E1-M2"]).get_data()
        e2 = raw.copy().pick_channels(["E2-M2"]).get_data()
        h_eog = e1 - e2

        raw.drop_channels(["E1-M2", "E2-M2"])

        eog_info = mne.create_info(["EOG"], raw.info["sfreq"], ch_types=["eog"])

        eog_raw = mne.io.RawArray(h_eog, eog_info)

        raw.add_channels([eog_raw], force_update_info=True)

    # Ensure that every input to the model meets same ordering of signal values
    final_order = ["C4-M1", "C3-M2", "EOG", "EMG"]
    raw.pick_channels(final_order)

    print(f"{sn} FINAL CHANNELS:", raw.ch_names)

    # Filter out noisy data, differnet thresholds depending on the given signal
    raw.filter(0.3, 35., picks=["C4-M1"])
    raw.filter(0.3, 35., picks=["C3-M2"])
    raw.filter(0.3, 10., picks=["EOG"])
    raw.filter(10., 100., picks=["EMG"])

    # Break data into 30 second epochs in input
    epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True)
    X = epochs.get_data()   # shape: (num_epochs, 4, 7680)

    # Extract expected labels and ensure input/output sizes align
    labels_raw = [stage_map.get(desc, None) for desc in ann.description]
    labels_raw = [l for l in labels_raw if l is not None]

    min_len = min(len(labels_raw), len(X))
    X = X[:min_len]
    y = np.array(labels_raw[:min_len])

    # Normalize data
    X = (X - X.mean(axis=-1, keepdims=True)) / X.std(axis=-1, keepdims=True)

    # Save the processed file with ".npz" extension
    np.savez(os.path.join(out_dir, f"{sn}.npz"), X=X, y=y)

    print(f"{sn} saved:", X.shape, y.shape)

def generate_subject_names(count):
    return [f"SN{i:03d}" for i in range(1, count + 1)]


if __name__ == "__main__":

    all_subjects = generate_subject_names(153)
    train_subjects, test_subjects = train_test_split(
        all_subjects,
        test_size=0.3, 
        random_state=42 
    )
    
    print(f"train_subjects size: {len(train_subjects)}")
    print(f"test_subjects size: {len(test_subjects)}")

    print("start processing train_subjects")
    for sn in train_subjects:
        process_subject(sn, out_dir="./processed_train_data") 

    print("start processing test_subjects")
    for sn in test_subjects:
        process_subject(sn, out_dir="./processed_test_data")

def openFile(file):
    data = np.load("processed/" + file)

    X_data = data['X']
    Y_data = data['y']

    print("X_data epochs:", len(X_data))
    print("Y_data labels:", len(Y_data))
    print(X_data.shape)
