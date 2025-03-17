import os
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch


def load_fif_data(data_dir):
    subject_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Limit to the first folder for testing
    #delete this later
    subject_folders = [subject_folders[0]]

    # Initialize empty lists to store the data and labels
    X = []  # To store EEG data
    Y = []  # To store labels

    # Read data from dataset 1
    for subject in subject_folders:
        subject_path = os.path.join(data_dir, subject)

        # List all EDF files for the subject
        fif_files = [f for f in os.listdir(subject_path)]

        for fif_file in fif_files:
            fif_path = os.path.join(subject_path, fif_file)


            print(f"Loading {fif_path}...")

            # Read the EDF file
            raw = mne.io.read_raw_fif(fif_path,preload=False)

            # Extract events and event_id from annotations (T0, T1, T2)
            events, event_id = mne.events_from_annotations(raw)

            # Create epochs with modified events and filtered_event_id
            epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=3, baseline=None, preload=True)
            #print(f"Used Annotations descriptions in epochs: {epochs.event_id}")

            # Process the EEG data as before
            epoch_data = epochs.get_data()  # EEG data (shape: n_epochs, n_channels, n_times)

            X.append(epoch_data)
            Y.append(epochs.events[:, -1])

            #print some general info about the data
            print(raw.info)
            print(f"Sampling frequency: {raw.info['sfreq']} Hz")
            print(f"Number of channels: {len(raw.ch_names)}")
            print(f"Channel names: {raw.ch_names}")
            print("Annotations:", raw.annotations)
            print("Extracted events:\n", events)
            print("Event ID mapping:", event_id)

    # Convert the list to numpy array
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # One-Hot Encode the Labels (T0, T1, T2, T3, T4)
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Y_train = encoder.fit_transform(Y_train.reshape(-1, 1))
    Y_test = encoder.transform(Y_test.reshape(-1, 1))

    # Convert one-hot-encoded labels to integer class labels
    Y_train = torch.argmax(torch.tensor(Y_train, dtype=torch.float32), axis=1).long()
    Y_test = torch.argmax(torch.tensor(Y_test, dtype=torch.float32), axis=1).long()

    return X_train, X_test, Y_train, Y_test

