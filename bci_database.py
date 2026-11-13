"""Build and read a local SQLite cache of labeled BCI IV 2a trials."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import mne
import numpy as np


CLASSES = {"769": "left_hand", "770": "right_hand", "771": "feet", "772": "tongue"}
CLASS_IDS = {name: i for i, name in enumerate(CLASSES.values())}


def encode_trial(trial):
    buffer = io.BytesIO()
    np.save(buffer, trial, allow_pickle=False)
    return buffer.getvalue()


def decode_trial(blob):
    return np.load(io.BytesIO(blob), allow_pickle=False)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_database(data_dir, output, tmin=0.0, tmax=4.0):
    """Create a new database; never overwrite an existing cache."""
    data_dir, output = Path(data_dir), Path(output)
    files = sorted(data_dir.glob("*T.gdf"))
    if not files:
        raise FileNotFoundError(f"No labeled *T.gdf files in {data_dir}")
    if output.exists():
        raise FileExistsError(f"Database already exists: {output}")
    if tmax <= tmin:
        raise ValueError("tmax must be greater than tmin")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    # Exclusive creation also prevents accidentally reusing an unfinished cache.
    with staging.open("xb"):
        pass
    config = {
        "converter_version": 1, "mne_version": mne.__version__,
        "highpass_hz": 2.0, "lowpass_hz": None,
        "tmin": tmin, "tmax": tmax, "baseline": None,
        "normalization": "none", "dtype": "float32",
        "session_selection": "labeled T.gdf only",
        "artifact_rejection": "no additional rejection; MNE annotation defaults",
    }
    try:
        with closing(sqlite3.connect(staging)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript("""
                CREATE TABLE preprocessing (
                    id INTEGER PRIMARY KEY,
                    config_json TEXT NOT NULL UNIQUE
                );
                CREATE TABLE recordings (
                    id INTEGER PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    session TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL
                );
                CREATE TABLE trials (
                    id INTEGER PRIMARY KEY,
                    recording_id INTEGER NOT NULL REFERENCES recordings(id),
                    preprocessing_id INTEGER NOT NULL REFERENCES preprocessing(id),
                    event_sample INTEGER NOT NULL,
                    event_description TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    sampling_rate REAL NOT NULL,
                    channel_names_json TEXT NOT NULL,
                    signal_npy BLOB NOT NULL,
                    UNIQUE(recording_id, preprocessing_id, event_sample)
                );
                CREATE INDEX trial_lookup
                    ON trials(preprocessing_id, recording_id, class_name);
            """)
            connection.execute("INSERT INTO preprocessing VALUES (1, ?)",
                               (json.dumps(config, sort_keys=True),))
            for path in files:
                print(f"Extracting {path.name} ...", flush=True)
                raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
                drop = [ch for ch in ("EOG-left", "EOG-central", "EOG-right")
                        if ch in raw.ch_names]
                if drop:
                    raw.drop_channels(drop)
                raw.filter(l_freq=2.0, h_freq=None, verbose="ERROR")
                events, event_ids = mne.events_from_annotations(raw, verbose="ERROR")
                selected = {desc: event_ids[desc] for desc in CLASSES if desc in event_ids}
                if not selected:
                    raise ValueError(f"No labeled motor imagery events in {path}")
                events = events[np.isin(events[:, 2], list(selected.values()))]
                epochs = mne.Epochs(raw, events, event_id=selected, tmin=tmin,
                                    tmax=tmax, baseline=None, preload=True, verbose="ERROR")
                data = epochs.get_data().astype(np.float32)
                if not len(data) or not np.isfinite(data).all():
                    raise ValueError(f"Empty or nonfinite EEG data in {path}")
                recording_id = connection.execute(
                    "INSERT INTO recordings(dataset, subject, session, source_path, source_sha256) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("BCICIV_2a", path.stem[:-1], "T", str(path.resolve()), sha256(path))
                ).lastrowid
                code_to_desc = {code: desc for desc, code in selected.items()}
                channels = json.dumps(epochs.ch_names)
                counts = Counter()
                for signal, event in zip(data, epochs.events):
                    desc = code_to_desc[int(event[2])]
                    blob = encode_trial(signal)
                    trial_id = connection.execute(
                        "INSERT INTO trials(recording_id, preprocessing_id, event_sample, "
                        "event_description, class_name, sampling_rate, channel_names_json, signal_npy) "
                        "VALUES (?, 1, ?, ?, ?, ?, ?, ?)",
                        (recording_id, int(event[0]), desc, CLASSES[desc],
                         float(raw.info["sfreq"]), channels, blob)
                    ).lastrowid
                    saved = connection.execute("SELECT signal_npy FROM trials WHERE id=?",
                                               (trial_id,)).fetchone()[0]
                    if not np.array_equal(signal, decode_trial(saved)):
                        raise ValueError(f"Trial {trial_id} failed round-trip verification")
                    counts[CLASSES[desc]] += 1
                connection.commit()
                print(f"  Verified {len(data)} trials, shape {data.shape[1:]}, "
                      f"classes {dict(counts)}; dropped {len(events) - len(data)}", flush=True)
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("SQLite foreign key check failed")
            total = connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        # On Windows rename fails if the destination already exists.
        staging.rename(output)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    print(f"Created {output.resolve()} with {total} verified trials "
          f"({output.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def load_bci_trials(database="datasets/bci_trials.sqlite", subjects=None):
    """Return X, y, channel names, and trial metadata, without splitting/normalizing.

    Labels: 0=left hand, 1=right hand, 2=feet, 3=tongue.
    Metadata preserves subject/session information for experiment splits.
    """
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        query = """SELECT t.id, r.subject, r.session, t.event_sample, t.class_name,
                          t.sampling_rate, t.channel_names_json, t.signal_npy
                   FROM trials t JOIN recordings r ON r.id=t.recording_id"""
        params = []
        if subjects is not None:
            params = list(subjects)
            if not params:
                raise ValueError("subjects must not be empty")
            query += " WHERE r.subject IN (" + ",".join("?" for _ in params) + ")"
        cursor = connection.execute(query + " ORDER BY t.id", params)
        signals, labels, metadata = [], [], []
        channel_names, sfreq = None, None
        for trial_id, subject, session, sample, label, rate, channels, blob in cursor:
            names = json.loads(channels)
            if channel_names is None:
                channel_names, sfreq = names, rate
            if channel_names != names or sfreq != rate:
                raise ValueError("Selected trials have incompatible channels or sampling rates")
            signals.append(decode_trial(blob))
            labels.append(CLASS_IDS[label])
            metadata.append({"trial_id": trial_id, "subject": subject, "session": session,
                             "event_sample": sample, "sampling_rate": rate})
    if not signals:
        raise ValueError("No matching trials in database")
    return np.stack(signals), np.asarray(labels, dtype=np.int64), channel_names, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/BCICIV_2a_gdf")
    parser.add_argument("--output", default="datasets/bci_trials.sqlite")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    args = parser.parse_args()
    build_database(args.data_dir, args.output, args.tmin, args.tmax)
