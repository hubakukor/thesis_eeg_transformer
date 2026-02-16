import numpy as np
from scipy.signal import stft
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import balanced_accuracy_score

def pick_channels_by_name(ch_names, wanted=("EEG-C3", "EEG-Cz", "EEG-C4")):
    name_to_idx = {name: i for i, name in enumerate(ch_names)}
    idxs = [name_to_idx[w] for w in wanted if w in name_to_idx]
    if len(idxs) != len(wanted):
        missing = [w for w in wanted if w not in name_to_idx]
        raise ValueError(f"Missing required channels: {missing}. Available: {ch_names}")
    return idxs

def eeg_to_logspec_stft(
    X,                 # np.ndarray [N, C, T]
    sfreq=250,
    ch_idxs=None,      # list[int] or None
    nperseg=128,
    noverlap=96,
    fmin=6.0,
    fmax=35.0,
    eps=1e-8,
):
    """Returns log-power spectrograms. Output: [N, C_sel, F, TT]."""
    if ch_idxs is None:
        ch_idxs = list(range(X.shape[1]))

    Xsel = X[:, ch_idxs, :]  # [N, Csel, T]
    N, Csel, T = Xsel.shape

    specs = []
    for i in range(N):
        trial_specs = []
        for c in range(Csel):
            f, tt, Zxx = stft(
                Xsel[i, c],
                fs=sfreq,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                boundary=None,
                padded=False,
            )
            P = (np.abs(Zxx) ** 2)
            band = (f >= fmin) & (f <= fmax)
            P = P[band, :]
            #start of freq pooling on whole trials
            k = 32  # 4 bins ~ 4 * 0.25Hz = 1Hz (if nperseg=1001)
            F, TT = P.shape
            F2 = (F // k) * k
            P = P[:F2, :]
            P = P.reshape(F2 // k, k, TT).mean(axis=1)

            #end

            P = np.log(P + eps).astype(np.float32)  # [F, TT]
            trial_specs.append(P)

        trial_specs = np.stack(trial_specs, axis=0)  # [Csel, F, TT]
        specs.append(trial_specs)

    return np.stack(specs, axis=0)  # [N, Csel, F, TT]


def evaluate_model_tf(model, X_tf, y, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y = y if torch.is_tensor(y) else torch.from_numpy(y).long()

    ds = TensorDataset(torch.from_numpy(X_tf).float(), y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    crit = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            total_loss += loss.item() * xb.size(0)

            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    avg_loss = total_loss / len(ds)
    acc = (preds == targets).mean()
    bal_acc = balanced_accuracy_score(targets, preds)
    return avg_loss, acc, bal_acc