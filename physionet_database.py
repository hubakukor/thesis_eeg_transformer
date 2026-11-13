"""Convert PhysioNet EEG movement/imagery EDF task runs to SQLite."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import mne
import numpy as np

from bci_database import encode_trial, decode_trial, sha256


CLASS_IDS = {"rest": 0, "left_hand": 1, "right_hand": 2, "both_fists": 3, "feet": 4}
IMAGERY_RUNS = {4, 6, 8, 10, 12, 14}
LEFT_RIGHT_RUNS = {3, 4, 7, 8, 11, 12}
DEFAULT_DATABASE = Path(__file__).resolve().parent / "datasets/physionet_trials.sqlite"


def build_database(data_dir, output=DEFAULT_DATABASE):
    data_dir, output = Path(data_dir), Path(output)
    files = sorted(p for p in data_dir.glob("S*/S*R*.edf")
                   if 3 <= int(p.stem.split("R")[-1]) <= 14)
    if not files:
        raise FileNotFoundError(f"No task EDF recordings in {data_dir}")
    if output.exists():
        raise FileExistsError(f"Database already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    with staging.open("xb"):
        pass
    config = {
        "converter_version": 1, "mne_version": mne.__version__,
        "highpass_hz": None, "lowpass_hz": None, "normalization": "none",
        "target_sfreq": 160.0, "window_seconds": [0.0, 3.0],
        "endpoint": "exclusive (480 samples)", "dtype": "float32", "units": "volts",
        "runs": "3-14; execution and imagery retained separately; baseline runs excluded",
        "artifact_rejection": "no additional rejection; MNE annotation defaults",
        "label_ids": CLASS_IDS,
    }
    totals = Counter()
    try:
        with closing(sqlite3.connect(staging)) as connection, connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript("""
                CREATE TABLE preprocessing(id INTEGER PRIMARY KEY, config_json TEXT NOT NULL UNIQUE);
                CREATE TABLE recordings(
                    id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, subject TEXT NOT NULL,
                    session TEXT NOT NULL, run INTEGER NOT NULL, task_type TEXT NOT NULL,
                    original_sampling_rate REAL NOT NULL,
                    source_path TEXT NOT NULL UNIQUE, source_sha256 TEXT NOT NULL,
                    event_count INTEGER NOT NULL, trial_count INTEGER NOT NULL,
                    dropped_epochs_json TEXT NOT NULL);
                CREATE TABLE trials(
                    id INTEGER PRIMARY KEY,
                    recording_id INTEGER NOT NULL REFERENCES recordings(id),
                    preprocessing_id INTEGER NOT NULL REFERENCES preprocessing(id),
                    event_sample INTEGER NOT NULL, event_description TEXT NOT NULL,
                    class_name TEXT NOT NULL, sampling_rate REAL NOT NULL,
                    channel_names_json TEXT NOT NULL, signal_npy BLOB NOT NULL,
                    UNIQUE(recording_id, preprocessing_id, event_sample));
                CREATE INDEX trial_lookup ON trials(preprocessing_id, recording_id, class_name);
                CREATE INDEX recording_lookup ON recordings(subject, task_type, run);
            """)
            connection.execute("INSERT INTO preprocessing VALUES(1, ?)",
                               (json.dumps(config, sort_keys=True),))
            for index, path in enumerate(files, 1):
                run = int(path.stem.split("R")[-1])
                task = "imagery" if run in IMAGERY_RUNS else "execution"
                raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
                original_rate = float(raw.info["sfreq"])
                if original_rate != 160.0:
                    raw.resample(160.0, verbose="ERROR")
                events, ids = mne.events_from_annotations(raw, verbose="ERROR")
                selected = {desc: ids[desc] for desc in ("T0", "T1", "T2") if desc in ids}
                if set(selected) != {"T0", "T1", "T2"}:
                    raise ValueError(f"Missing task annotations in {path}")
                events = events[np.isin(events[:, 2], list(selected.values()))]
                # Inclusive MNE endpoint is the last sample in [0, 3 seconds).
                epochs = mne.Epochs(raw, events, selected, tmin=0, tmax=479 / 160,
                                    baseline=None, preload=True, verbose="ERROR")
                data = epochs.get_data().astype(np.float32)
                if data.shape[1:] != (64, 480) or not len(data) or not np.isfinite(data).all():
                    raise ValueError(f"Invalid trials in {path}: {data.shape}")
                dropped = [{"event_index": i, "reasons": list(reasons)}
                           for i, reasons in enumerate(epochs.drop_log) if reasons]
                recording_id = connection.execute(
                    "INSERT INTO recordings(dataset, subject, session, run, task_type, "
                    "original_sampling_rate, source_path, source_sha256, event_count, trial_count, "
                    "dropped_epochs_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("PhysioNet_EEGMMIDB", path.parent.name, "unspecified", run, task,
                     original_rate, str(path.resolve()), sha256(path), len(events), len(data),
                     json.dumps(dropped))).lastrowid
                mapping = {"T0": "rest", "T1": "left_hand" if run in LEFT_RIGHT_RUNS else "both_fists",
                           "T2": "right_hand" if run in LEFT_RIGHT_RUNS else "feet"}
                reverse = {code: desc for desc, code in selected.items()}
                channels = json.dumps(epochs.ch_names)
                for signal, event in zip(data, epochs.events):
                    desc = reverse[int(event[2])]
                    label = mapping[desc]
                    trial_id = connection.execute(
                        "INSERT INTO trials(recording_id, preprocessing_id, event_sample, "
                        "event_description, class_name, sampling_rate, channel_names_json, signal_npy) "
                        "VALUES(?, 1, ?, ?, ?, 160, ?, ?)",
                        (recording_id, int(event[0]), desc, label, channels, encode_trial(signal))
                    ).lastrowid
                    saved = connection.execute("SELECT signal_npy FROM trials WHERE id=?", (trial_id,)).fetchone()[0]
                    if not np.array_equal(signal, decode_trial(saved)):
                        raise ValueError(f"Round-trip verification failed: trial {trial_id}")
                    totals[label] += 1
                connection.commit()
                if index % 12 == 0 or index == len(files):
                    print(f"Verified {index}/{len(files)} recordings; {sum(totals.values())} trials "
                          f"(latest: {path.name})", flush=True)
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Foreign key check failed")
        staging.rename(output)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    print(f"Created {output.resolve()}: {dict(totals)}; "
          f"{output.stat().st_size / 1024**3:.2f} GiB", flush=True)


def load_physionet_trials(database=DEFAULT_DATABASE, subjects=None, task_type=None):
    """Return X, y, channels, metadata. Filter before loading large arrays into RAM.

    task_type: 'imagery', 'execution', or None (both).
    Labels: rest=0, left=1, right=2, both fists=3, feet=4.
    """
    if task_type not in (None, "imagery", "execution"):
        raise ValueError("task_type must be imagery, execution, or None")
    conditions, params = [], []
    if subjects is not None:
        subjects = list(subjects)
        if not subjects:
            raise ValueError("subjects must not be empty")
        conditions.append("r.subject IN (" + ",".join("?" for _ in subjects) + ")")
        params.extend(subjects)
    if task_type:
        conditions.append("r.task_type=?")
        params.append(task_type)
    query = """SELECT t.id, r.subject, r.run, r.task_type, t.event_sample,
                      t.class_name, t.sampling_rate, t.channel_names_json, t.signal_npy
               FROM trials t JOIN recordings r ON r.id=t.recording_id"""
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM (" + query + ")", params).fetchone()[0]
        if not count:
            raise ValueError("No matching trials")
        X = np.empty((count, 64, 480), dtype=np.float32)
        y = np.empty(count, dtype=np.int64)
        metadata, channels = [], None
        for i, row in enumerate(connection.execute(query + " ORDER BY t.id", params)):
            trial_id, subject, run, task, sample, label, rate, names, blob = row
            names = json.loads(names)
            if channels is None:
                channels = names
            if names != channels or rate != 160:
                raise ValueError("Incompatible channel order or sampling rate")
            X[i] = decode_trial(blob)
            y[i] = CLASS_IDS[label]
            metadata.append({"trial_id": trial_id, "subject": subject, "run": run,
                             "task_type": task, "event_sample": sample, "sampling_rate": rate})
    return X, y, channels, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent /
                        "datasets/eeg-motor-movementimagery-dataset-1.0.0/files")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    build_database(args.data_dir, args.output)
