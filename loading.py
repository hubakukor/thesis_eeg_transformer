import os
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
from scipy.stats import zscore

#normalize the eeg signal between -1 and 1
def normalize_epoch_minmax(epoch):
    """
    Normalizes the EEG signal between -1 and 1 using the minimum and maximum values of each channel.

    Args:
        epoch (np.array): A numpy array containing the EEG data.

    Returns:
        np.array: A numpy array containing the normalized EEG data.
    """
    min_vals = epoch.min(axis=1, keepdims=True)
    max_vals = epoch.max(axis=1, keepdims=True)
    scale = max_vals - min_vals
    scale[scale == 0] = 1e-8  # avoid dividing by zero
    norm = 2 * (epoch - min_vals) / scale - 1  # scales to [-1, 1]
    return norm


def load_fif_data(data_dir, event_id, test_set=False):
    """
    Loads the data and splits it into training, validation, and test sets.

    Args:
        data_dir (str): The directory containing the EEG data.
        event_id (dict): A dictionary mapping event names to their corresponding integer IDs.
        test_set (bool): Whether to create the test set or not.

    Returns:
        X_train, X_val, X_test (optional), Y_train, Y_val, Y_test (optional): Lists of numpy arrays containing the training, validation, and test data, and their corresponding labels.
    """

    subject_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Limit to the first folder for testing
    #subject_folders = [subject_folders[0]]

    # Initialize empty lists to store the data and labels
    X = []  # To store EEG data
    Y = []  # To store labels

    # Read data
    for subject in subject_folders:
        subject_path = os.path.join(data_dir, subject)

        # List all EDF files for the subject
        fif_files = [f for f in os.listdir(subject_path)]

        for fif_file in fif_files:
            fif_path = os.path.join(subject_path, fif_file)


            print(f"Loading {fif_path}...")

            # Read the FIF file
            raw = mne.io.read_raw_fif(fif_path,preload=True)
            #print('Annotations: ',raw.annotations)
            #print('Annotation duration: ',raw.annotations.duration)

            #bandpass filter
            # raw.filter(l_freq=5, h_freq=30)
            raw.filter(l_freq=2, h_freq=None)

            # Extract events and event_id from annotations
            events, extracted_event_id = mne.events_from_annotations(raw)

            # Map event names (e.g. 'Stimulus/13') to extracted IDs
            filtered_event_id = {k: extracted_event_id[k] for k in event_id.keys() if k in extracted_event_id}


            # Filter only relevant event types
            filtered_events = [e for e in events if e[-1] in filtered_event_id.values()]
            events = np.array(filtered_events)
            if len(events) == 0:
                print(f"No matching events found in {fif_file}, skipping.")
                continue
            # Create epochs with modified events and filtered_event_id
            epochs = mne.Epochs(raw, events, filtered_event_id, tmin=0, tmax=3, baseline=None, preload=True)
            #print(f"Used Annotations descriptions in epochs: {epochs.event_id}")

            # Process the EEG data
            epoch_data = epochs.get_data()  # EEG data (shape: n_epochs, n_channels, n_times)

            # Normalize each epoch between -1 and 1
            # epoch_data = np.array([normalize_epoch_minmax(e) for e in epoch_data])

            # Apply z-score normalization to each channel
            epoch_data = np.array([np.nan_to_num(zscore(e, axis=1)) for e in epoch_data])

            # Map back to 0/1 using your event_id dictionary
            label_map = {v: event_id[k] for k, v in filtered_event_id.items()}
            Y.append(np.vectorize(label_map.get)(epochs.events[:, -1]))

            X.append(epoch_data)


            # #print some general info about the data
            # print(f"Number of channels: {len(raw.ch_names)}")
            # print(raw.info)
            # print(f"Sampling frequency: {raw.info['sfreq']} Hz")
            # print(f"Number of channels: {len(raw.ch_names)}")
            # print(f"Channel names: {raw.ch_names}")
            # print("Annotations:", raw.annotations)
            # print("Extracted events:\n", events)
            # print("Event ID mapping:", event_id)
            # unique, counts = np.unique(events[:, -1], return_counts=True)
            # print("Filtered Event Frequency:", dict(zip(unique, counts)))
            # print("Final unique labels in Y:", np.unique(Y))



    # Convert the list to numpy array
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)


    if test_set:
        # First split into train, validation and test sets (0.64, 0.16, 0.20)
        X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)
    else:
        # First split into train and validation sets (0.80, 0.20)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)




    # One-Hot Encode the Labels
    encoder = OneHotEncoder(sparse_output=False)
    Y_train = encoder.fit_transform(Y_train.reshape(-1, 1))
    if test_set:
        Y_test = encoder.transform(Y_test.reshape(-1, 1))
    Y_val = encoder.transform(Y_val.reshape(-1, 1))

    # Convert one-hot-encoded labels to integer class labels
    Y_train = torch.argmax(torch.tensor(Y_train, dtype=torch.float32), axis=1).long()
    if test_set:
        Y_test = torch.argmax(torch.tensor(Y_test, dtype=torch.float32), axis=1).long()
    Y_val = torch.argmax(torch.tensor(Y_val, dtype=torch.float32), axis=1).long()

    #check label distribution
    unique, counts = np.unique(Y, return_counts=True)
    print("Final label distribution (all data):", dict(zip(unique, counts)))

    if test_set:
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    else:
        return X_train, X_val, Y_train, Y_val

def load_for_complete_cross_validation(data_dir, event_id, test_folder, augment_train_ratio = 0):
    """
    Loads the data and splits it into training, validation, and test sets.
    The specified test folder will be used for testing, the rest is for training and validation.

    Args:
        data_dir (str): The directory containing the EEG data.
        event_id (dict): A dictionary mapping event names to their corresponding integer IDs.
        test_folder (str): The name of the folder containing the test data.
        augment_train_ratio (float): The ratio of augmented samples to create. 0 means no augmentation.

    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test: Lists of numpy arrays containing the training, validation, and test data, and their corresponding labels.
    """
    # Get a list of all subject folders
    subject_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]


    # Separate the folders from the test folder
    train_val_folders = [f for f in subject_folders if f != test_folder]

    def load_data_from_folders(folders):
        """
        Loads the data from the specified folders and combines them.

        args:
            folders (list): A list of folder names.

        returns:
            X (np.array): A numpy array containing the EEG data.
            Y (np.array): A numpy array containing the labels.
        """
        X, Y = [], []
        for subject in folders:
            subject_path = os.path.join(data_dir, subject)
            fif_files = [f for f in os.listdir(subject_path)]

            for fif_file in fif_files:
                fif_path = os.path.join(subject_path, fif_file)
                print(f"Loading {fif_path}...")

                raw = mne.io.read_raw_fif(fif_path, preload=True)
                raw.filter(l_freq=2, h_freq=None) #apply high-pass filter

                events, extracted_event_id = mne.events_from_annotations(raw)

                # Filter events based on the event_id dictionary
                filtered_event_id = {k: extracted_event_id[k] for k in event_id if k in extracted_event_id}
                filtered_events = [e for e in events if e[-1] in filtered_event_id.values()]
                if len(filtered_events) == 0:
                    print(f"No matching events in {fif_file}, skipping.")
                    continue

                epochs = mne.Epochs(raw, np.array(filtered_events), filtered_event_id, tmin=0, tmax=3, baseline=None, preload=True)
                epoch_data = epochs.get_data()

                # Normalize each epoch between -1 and 1
                epoch_data = np.array([normalize_epoch_minmax(e) for e in epoch_data])

                # Apply z-score normalization to each channel in each epoch
                # epoch_data = np.array([np.nan_to_num(zscore(e, axis=1)) for e in epoch_data])

                label_map = {v: event_id[k] for k, v in filtered_event_id.items()}
                labels = np.vectorize(label_map.get)(epochs.events[:, -1])

                X.append(epoch_data)
                Y.append(labels)

        # Return an empty array with the correct shape if no data was loaded
        if len(X) == 0:
            return np.empty((0, 63, 1501)), np.empty((0,))
        return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

    # Load test and train/val data
    X_test, Y_test = load_data_from_folders([test_folder])
    X_trainval, Y_trainval = load_data_from_folders(train_val_folders)

    # Split training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42)

    if augment_train_ratio != 0:
        print('augmenting training data by mixing segments')
        X_train, Y_train = augment_train_by_mixing(X_train, Y_train, aug_factor=augment_train_ratio)

    # One-hot encoding and conversion to class indices
    encoder = OneHotEncoder(sparse_output=False)
    Y_all = np.concatenate([Y_train, Y_val, Y_test]).reshape(-1, 1)
    encoder.fit(Y_all)

    Y_train = encoder.transform(Y_train.reshape(-1, 1))
    Y_val = encoder.transform(Y_val.reshape(-1, 1))
    Y_test = encoder.transform(Y_test.reshape(-1, 1))

    Y_train = torch.argmax(torch.tensor(Y_train, dtype=torch.float32), dim=1).long()
    Y_val = torch.argmax(torch.tensor(Y_val, dtype=torch.float32), dim=1).long()
    Y_test = torch.argmax(torch.tensor(Y_test, dtype=torch.float32), dim=1).long()

    print("Label distribution:")
    print("Train:", dict(zip(*np.unique(Y_train.numpy(), return_counts=True))))
    print("Val:  ", dict(zip(*np.unique(Y_val.numpy(), return_counts=True))))
    print("Test: ", dict(zip(*np.unique(Y_test.numpy(), return_counts=True))))

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def augment_train_by_mixing(X, y, aug_factor=1.0, cut_range=(0.35, 0.65), seed=42):
    """
    Augment the training data by mixing segments of the real trials to create artificial trials.

    X: np.ndarray [N, C, T]
    y: np.ndarray [N] (integer class labels)
    aug_factor: artificial/real ratio
    cut_range: The point at which to cut a trial is randomly chosen in this range.
    """
    assert X.ndim == 3, "X must be [N, C, T]"
    N, C, T = X.shape
    rng = np.random.default_rng(seed)

    n_aug = int(round(N * float(aug_factor))) # number of augmented samples to create
    X_aug = np.empty((n_aug, C, T), dtype=X.dtype)
    y_aug = np.empty((n_aug,), dtype=y.dtype)

    # indices per class for quick same-class sampling
    classes = np.unique(y)
    idx_per_class = {c: np.flatnonzero(y == c) for c in classes}

    for i in range(n_aug):
        # choose a class to generate from
        c = rng.choice(classes)
        idxs = idx_per_class[c]

        # pick two trials from that class
        if len(idxs) >= 2:
            i1, i2 = rng.choice(idxs, size=2, replace=False)
        else:
            i1 = i2 = idxs[0]

        x1, x2 = X[i1], X[i2]

        # choose a cut point
        r = float(rng.uniform(cut_range[0], cut_range[1]))   #choose the cut ratio
        cut = int(np.clip(round(T * r), 1, T - 1))  # exact cut point, ensure the cut point is inside the actual points-1

        # new trial: beginning of x1 + end of x2 (same class → safe label)
        X_aug[i] = np.concatenate([x1[:, :cut], x2[:, cut:]], axis=1)
        y_aug[i] = c

    # combine and shuffle
    X_all = np.concatenate([X, X_aug], axis=0)
    y_all = np.concatenate([y, y_aug], axis=0)
    perm = rng.permutation(X_all.shape[0]) #randomly shuffle the training trials
    return X_all[perm], y_all[perm]


def load_bci_dataset(data_dir, tmin=0.0, tmax=4.0):
    """
    Loads BCI Competition IV 2a dataset (both training and evaluation GDF files)

    Args:
        data_dir (str): Folder containing the A0xT.gdf and A0xE.gdf files.
        tmin (float): Start time of epochs (in seconds).
        tmax (float): End time of epochs (in seconds).

    Returns:
        X_train, Y_train, X_test, Y_test, X_val, Y_val : numpy arrays
    """
    motor_imagery_event_ids = {
        "left_hand": 769,
        "right_hand": 770,
        "feet": 771,
        "tongue": 772,
    }

    X_all, Y_all = [], []

    for file in os.listdir(data_dir):
        if not file.endswith("T.gdf"):
            continue

        file_path = os.path.join(data_dir, file)
        print(f"\nProcessing file: {file_path}")
        raw = mne.io.read_raw_gdf(file_path, preload=True)

        # Drop EOG channels
        for ch in ['EOG-left', 'EOG-central', 'EOG-right']:
            if ch in raw.ch_names:
                raw.drop_channels([ch])

        # High-pass filter
        raw.filter(l_freq=2.0, h_freq=None)

        # Extract events
        events, event_id = mne.events_from_annotations(raw)
        motor_imagery_ids = [event_id.get(str(val)) for val in motor_imagery_event_ids.values()]
        motor_imagery_ids = [e for e in motor_imagery_ids if e is not None]

        if not motor_imagery_ids:
            print(f"No valid motor imagery IDs in {file}")
            continue

        # Filter only MI events
        mi_events = np.array([e for e in events if e[2] in motor_imagery_ids])
        if len(mi_events) == 0:
            print(f"No MI events in {file}")
            continue

        # Epoching
        epochs = mne.Epochs(
            raw, mi_events,
            {key: event_id.get(str(val)) for key, val in motor_imagery_event_ids.items()},
            tmin=tmin, tmax=tmax, baseline=None, preload=True
        )
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        labels = epochs.events[:, -1]

        # Normalize each epoch
        data = np.array([normalize_epoch_minmax(e) for e in data])

        # Map event codes (7–10) → 1–4
        label_mapping = {7: 0, 8: 1, 9: 2, 10: 3}
        labels = np.array([label_mapping.get(y, -1) for y in labels])
        valid = labels != -1
        data, labels = data[valid], labels[valid]

        X_all.append(data)
        Y_all.append(labels)

    # Concatenate all subjects
    if X_all:
        X_all = np.concatenate(X_all, axis=0)
        Y_all = np.concatenate(Y_all, axis=0)
    else:
        print("No valid data found.")
        return (np.empty((0,)),)*6

    # Split into train/val/test: 80/10/10
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_all, Y_all, test_size=0.2, random_state=42, stratify=Y_all)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

    # Convert labels to torch tensors
    Y_train = torch.from_numpy(Y_train).long()
    Y_val = torch.from_numpy(Y_val).long()
    Y_test = torch.from_numpy(Y_test).long()

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    print("Label distribution:")
    print(f"  Train: {dict(zip(*np.unique(Y_train.numpy(), return_counts=True)))}")
    print(f"  Val:   {dict(zip(*np.unique(Y_val.numpy(), return_counts=True)))}")
    print(f"  Test:  {dict(zip(*np.unique(Y_test.numpy(), return_counts=True)))}")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def fix_epoch_length(epoch_data, target_samples):
    """Pad or trim epochs to the target number of samples."""
    if epoch_data.shape[-1] > target_samples:
        return epoch_data[:, :, :target_samples]
    elif epoch_data.shape[-1] < target_samples:
        padding = target_samples - epoch_data.shape[-1]
        return np.pad(epoch_data, ((0, 0), (0, 0), (0, padding)), mode='constant')
    return epoch_data


def make_default_split(subjects, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """Produces default 80/10/10 split if user provides none."""
    rng = np.random.RandomState(random_state)
    subjects = sorted(subjects)
    perm = rng.permutation(len(subjects))
    subjects = [subjects[i] for i in perm]

    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_subj = subjects[:n_train]
    val_subj   = subjects[n_train:n_train + n_val]
    test_subj  = subjects[n_train + n_val:]

    return train_subj, val_subj, test_subj

def load_physionet_eeg(
    data_dir,
    train_subjects=None,
    val_subjects=None,
    test_subjects=None,
    t_min=0.0,
    t_max=3.0
):
    """
    Load PhysioNet MI dataset with optional default subject split.

    Final labels:
        0 = rest
        1 = left hand
        2 = right hand
        3 = both fists
        4 = both feet

    Args:
        data_dir (str): Path to the PhysioNet EEG dataset folder.
        train_subjects (list): List of subject folders to use for training.
        val_subjects (list): List of subject folders to use for validation.
        test_subjects (list): List of subject folders to use for testing.
        t_min (float): Start time of epochs (in seconds).
        t_max (float): End time of epochs (in seconds).

    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """

    # Runs where T1/T2 mean both-fists / both-feet
    BOTH_LIMBS_RUNS = [5, 6, 9, 10, 13, 14]
    # Runs where T1 = left, T2 = right
    LEFT_RIGHT_RUNS = [3, 4, 7, 8, 11, 12]

    # Known nominal sampling rate for PhysioNet EEG (EEG channels)
    SFREQ_TARGET = 160.0
    target_samples = int((t_max - t_min) * SFREQ_TARGET)

    # Get list of subjects (folders)
    all_subjects = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])


    if len(all_subjects) == 0:
        raise RuntimeError("No subject folders found in PhysioNet dataset.")

    # If no subjects given → compute default split
    if train_subjects is None or val_subjects is None or test_subjects is None:
        train_subjects, val_subjects, test_subjects = make_default_split(all_subjects)

        print(f"Using default subject split:")
        print(f" Train: {train_subjects}")
        print(f" Val:   {val_subjects}")
        print(f" Test:  {test_subjects}")

    def load_from_subjects(subject_folders):
        X_list, Y_list = [], []

        for subject in subject_folders:
            subject_path = os.path.join(data_dir, subject)
            if not os.path.isdir(subject_path):
                print(f"Warning: subject folder not found: {subject_path}")
                continue

            edf_files = [f for f in os.listdir(subject_path) if f.endswith('.edf')]

            for edf_file in edf_files:
                edf_path = os.path.join(subject_path, edf_file)

                # Parse run number (R01 → 1)
                try:
                    run_number = int(edf_file.split("R")[1].split(".")[0])
                except Exception:
                    print(f"Skipping {edf_file}, cannot parse run number.")
                    continue

                print(f"Loading {edf_path} (run {run_number})...")

                raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel='auto', verbose="ERROR")

                sfreq = raw.info["sfreq"]
                if abs(sfreq - SFREQ_TARGET) > 1e-3:
                    print(f"Warning: sfreq={sfreq} Hz in {edf_file}, expected {SFREQ_TARGET} Hz. "
                          f"Will pad/trim to {target_samples} samples without resampling.")

                # Extract events
                events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

                if not {"T0", "T1", "T2"}.issubset(event_id.keys()):
                    print(f"Missing T0/T1/T2 in {edf_file}, skipping.")
                    continue

                code_T0, code_T1, code_T2 = event_id["T0"], event_id["T1"], event_id["T2"]

                # Epochs
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id={"T0": code_T0, "T1": code_T1, "T2": code_T2},
                    tmin=t_min,
                    tmax=t_max,
                    baseline=None,
                    preload=True,
                    verbose="ERROR"
                )

                epoch_data = epochs.get_data()
                labels_raw = epochs.events[:, -1]

                # Assign final semantic labels
                labels = np.full_like(labels_raw, -1)

                # T0 = rest
                labels[labels_raw == code_T0] = 0

                if run_number in BOTH_LIMBS_RUNS:
                    labels[labels_raw == code_T1] = 3
                    labels[labels_raw == code_T2] = 4
                elif run_number in LEFT_RIGHT_RUNS:
                    labels[labels_raw == code_T1] = 1
                    labels[labels_raw == code_T2] = 2
                else:
                    print(f"Unknown run {run_number}, skipping.")
                    continue

                valid = labels != -1
                if not np.any(valid):
                    continue

                epoch_data = epoch_data[valid]
                labels = labels[valid]

                # Enforce *global* target_samples for all data
                epoch_data = fix_epoch_length(epoch_data, target_samples)
                epoch_data = np.array([normalize_epoch_minmax(e) for e in epoch_data])

                X_list.append(epoch_data)
                Y_list.append(labels)

        if not X_list:
            return np.empty((0,)), np.empty((0,))

        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        return X, Y

    # Load each set
    X_train, Y_train = load_from_subjects(train_subjects)
    X_val, Y_val = load_from_subjects(val_subjects)
    X_test, Y_test = load_from_subjects(test_subjects)

    # To torch labels
    Y_train = torch.from_numpy(Y_train).long()
    Y_val = torch.from_numpy(Y_val).long()
    Y_test = torch.from_numpy(Y_test).long()

    print("\nFinal dataset shapes:")
    print(f" Train: {X_train.shape}, labels: {Y_train.shape}")
    print(f" Val:   {X_val.shape},   labels: {Y_val.shape}")
    print(f" Test:  {X_test.shape},  labels: {Y_test.shape}")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test