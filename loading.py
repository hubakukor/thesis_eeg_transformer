import os
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch

#normalize the eeg signal betwen -1 and 1
def normalize_epoch_minmax(epoch):
    min_vals = epoch.min(axis=1, keepdims=True)
    max_vals = epoch.max(axis=1, keepdims=True)
    scale = max_vals - min_vals
    scale[scale == 0] = 1e-8  # avoid dividing by zero
    norm = 2 * (epoch - min_vals) / scale - 1  # scales to [-1, 1]
    return norm


def load_fif_data(data_dir, event_id):
    subject_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Limit to the first folder for testing
    #delete this later
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
            epoch_data = np.array([normalize_epoch_minmax(e) for e in epoch_data])

            # Map back to 0/1 using your event_id dictionary
            label_map = {v: event_id[k] for k, v in filtered_event_id.items()}
            Y.append(np.vectorize(label_map.get)(epochs.events[:, -1]))

            X.append(epoch_data)


            print(f"Number of channels: {len(raw.ch_names)}")
            '''
            #print some general info about the data
            print(raw.info)
            print(f"Sampling frequency: {raw.info['sfreq']} Hz")
            print(f"Number of channels: {len(raw.ch_names)}")
            print(f"Channel names: {raw.ch_names}")
            print("Annotations:", raw.annotations)
            print("Extracted events:\n", events)
            print("Event ID mapping:", event_id)
            unique, counts = np.unique(events[:, -1], return_counts=True)
            print("Filtered Event Frequency:", dict(zip(unique, counts)))
            print("Final unique labels in Y:", np.unique(Y))
            '''


    # Convert the list to numpy array
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # One-Hot Encode the Labels
    encoder = OneHotEncoder(sparse_output=False)
    Y_train = encoder.fit_transform(Y_train.reshape(-1, 1))
    Y_test = encoder.transform(Y_test.reshape(-1, 1))

    # Convert one-hot-encoded labels to integer class labels
    Y_train = torch.argmax(torch.tensor(Y_train, dtype=torch.float32), axis=1).long()
    Y_test = torch.argmax(torch.tensor(Y_test, dtype=torch.float32), axis=1).long()

    #check label distribution
    unique, counts = np.unique(Y, return_counts=True)
    print("Final label distribution (all data):", dict(zip(unique, counts)))

    return X_train, X_test, Y_train, Y_test

