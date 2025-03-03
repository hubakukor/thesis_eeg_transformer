import os
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch



# Target number of samples (3 seconds at 160 Hz)
target_samples = 480
num_channels = 64  # EEG channels

# Function to pad or trim epochs to the target number of samples
def fix_epoch_length(epoch_data, target_samples):
    if epoch_data.shape[-1] > target_samples:
        # If the epoch is longer, trim it
        return epoch_data[:, :, :target_samples]
    elif epoch_data.shape[-1] < target_samples:
        # If the epoch is shorter, pad it with zeros
        padding = target_samples - epoch_data.shape[-1]
        return np.pad(epoch_data, ((0, 0), (0, 0), (0, padding)), mode='constant')
    else:
        # If the epoch has the correct length, return as is
        return epoch_data



# Path to folders with the data in dataset 1
data_dir = r"D:\suli\önlab\eeg-motor-movementimagery-dataset-1.0.0\files"

def load_edf_data(data_dir):
    subject_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Runs where both fists and both feet are involved
    target_runs = [5, 6, 9, 10, 13, 14]

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
        edf_files = [f for f in os.listdir(subject_path) if f.endswith('.edf')]

        for edf_file in edf_files:
            edf_path = os.path.join(subject_path, edf_file)

            # Identify run number to check if it's a target run
            run_number = int(edf_file.split('R')[1].split('.')[0])

            print(f"Loading {edf_path} for run {run_number} ...")

            # Read the EDF file
            raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel='auto')

            # Extract events and event_id from annotations (T0, T1, T2)
            events, event_id = mne.events_from_annotations(raw)

            # Modify event labels based on target vs. non-target runs
            if run_number in target_runs:
                # For target runs, map T1 to T3 (both fists) and T2 to T4 (both feet)
                if "T1" in event_id:
                    events[events[:, -1] == event_id["T1"], -1] = 4  # T1 -> T3
                if "T2" in event_id:
                    events[events[:, -1] == event_id["T2"], -1] = 5  # T2 -> T4

                # Update event_id dictionary to reflect new labels
                event_id = {"T0": 1, "T3": 4, "T4": 5}
                #print(f"Modified events for target run {run_number}: {events[:, -1]}")
            else:
                # Non-target runs: retain T0, T1, T2 labels
                event_id = {"T0": 1, "T1": 2, "T2": 3}

            # Filter event_id to include only present events in `events`
            filtered_event_id = {key: event_id[key] for key in event_id if event_id[key] in events[:, -1]}

            # Create epochs with modified events and filtered_event_id
            epochs = mne.Epochs(raw, events, filtered_event_id, tmin=0, tmax=3, baseline=None, preload=True)
            #print(f"Used Annotations descriptions in epochs: {epochs.event_id}")

            # Process the EEG data as before
            epoch_data = epochs.get_data()  # EEG data (shape: n_epochs, n_channels, n_times)
            epoch_data_fixed = fix_epoch_length(epoch_data, target_samples)

            X.append(epoch_data_fixed)  # Append fixed-length EEG data
            Y.append(epochs.events[:, -1])  # Event labels (1 = T0, 2 = T1, 3 = T2, 4 = T3, 5 = T4)

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

X_train, X_test, Y_train, Y_test = load_edf_data(data_dir)

#Second dataset (22 channels)
gdf_file_path = r"D:\suli\önlab\BCICIV_2a_gdf"


def load_training_files(data_folder, tmin=-0.2, tmax=4):
    motor_imagery_event_ids = {
        "left_hand": 769,
        "right_hand": 770,
        "feet": 771,
        "tongue": 772,
    }
    all_data = []
    all_labels = []

    for file in os.listdir(data_folder):
        if file.endswith("T.gdf"):
            file_path = os.path.join(data_folder, file)
            print(f"\nProcessing file: {file_path}")

            # Load the GDF file
            raw = mne.io.read_raw_gdf(file_path, preload=True)

            # Drop EOG channels
            raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

            # Extract events and map them using the event_id dictionary
            events, event_id = mne.events_from_annotations(raw)

            # Debug: Print event_id dictionary
            print(f"Event ID mapping: {event_id}")

            # Map motor imagery event IDs to their codes in the file
            motor_imagery_ids = [event_id.get(str(val)) for val in motor_imagery_event_ids.values()]
            motor_imagery_ids = [e for e in motor_imagery_ids if e is not None]  # Remove None values

            # Debug: Print mapped motor imagery IDs
            print(f"Mapped motor imagery event IDs: {motor_imagery_ids}")

            if not motor_imagery_ids:
                print(f"No valid motor imagery event IDs found in {file_path}. Skipping.")
                continue

            # Filter events for motor imagery tasks
            motor_imagery_events = np.array([e for e in events if e[2] in motor_imagery_ids])
            if len(motor_imagery_events) == 0:
                print(f"No motor imagery events found in {file_path}. Skipping.")
                continue

            print(f"Found {len(motor_imagery_events)} motor imagery events in {file_path}")

            # Create epochs for motor imagery events
            epochs = mne.Epochs(
                raw, motor_imagery_events,
                {key: event_id.get(str(val)) for key, val in motor_imagery_event_ids.items()},
                tmin=tmin, tmax=tmax, baseline=None, preload=True
            )

            # Append data and labels
            all_data.append(epochs.get_data())
            all_labels.append(epochs.events[:, -1])

    if len(all_data) == 0 or len(all_labels) == 0:
        print("No valid motor imagery data found in the dataset.")
        return np.empty((0,)), np.empty((0,))

    # Concatenate all data and labels
    X = np.concatenate(all_data, axis=0)
    Y = np.concatenate(all_labels, axis=0)

    # Debug: Print unique labels before mapping
    print(f"Unique labels before mapping: {np.unique(Y)}")

    # Map event codes to labels
    label_mapping = {7: 1, 8: 2, 9: 3, 10: 4}
    Y = np.array([label_mapping.get(y, -1) for y in Y])  # Map to labels or set to -1 for invalid labels

    # Filter out invalid labels
    valid_indices = Y != -1
    X = X[valid_indices]
    Y = Y[valid_indices]

    # Debug: Print unique labels after mapping
    print(f"Unique labels after mapping: {np.unique(Y)}")

    return X, Y



# Load the training data for the second dataset
X2_10_20, Y2 = load_training_files(gdf_file_path)

# # Check the results
# print(f"Data shape: {X2.shape}")  # (n_samples, n_channels, n_times)
# print(f"Labels shape: {Y2.shape}")  # (n_samples,)
# print(f"Sample labels: {Y2[:10]}")



#Convert the second dataset into 10-10 format
def reformat_to_10_10_system(data, current_channels, target_channels):
    """
    Reformat data to match the 10-10 system by adding missing channels filled with zeros.

    Parameters:
        data (np.ndarray): EEG data of shape (n_epochs, n_channels, n_times).
        current_channels (list): List of current channel names (10-20 system).
        target_channels (list): List of target channel names (10-10 system).

    Returns:
        np.ndarray: Reformatted EEG data of shape (n_epochs, len(target_channels), n_times).
    """
    # Create a mapping of channel indices for the current dataset
    channel_mapping = {ch: i for i, ch in enumerate(current_channels)}

    # Initialize new data array with zeros
    n_epochs, _, n_times = data.shape
    reformatted_data = np.zeros((n_epochs, len(target_channels), n_times))

    # Copy data for existing channels
    for i, ch in enumerate(target_channels):
        if ch in channel_mapping:
            reformatted_data[:, i, :] = data[:, channel_mapping[ch], :]

    return reformatted_data

# Define the 64-channel target (10-10 system)
target_channels = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.',
    'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..',
    'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.',
    'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']

#Channels from the smaller dataset (10-20 system)
#Using the channel names from dataset1, the order shows which channels point to which electrode
current_channels = ['Fz..', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.',
                    'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
                    'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.',
                    'P1..', 'Pz..', 'P2..', 'Poz.']

# Reformat the smaller dataset to match the 10-10 system
X2 = reformat_to_10_10_system(X2_10_20, current_channels, target_channels)