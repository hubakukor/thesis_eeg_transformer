"""Shared training runner and persistent local experiment history."""

import argparse
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import psutil

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "experiment_runs"
HISTORY = RUNS_DIR / "experiments.sqlite"
CLASSES = {
    "bci": ["left hand", "right hand", "feet", "tongue"],
    "physionet": ["rest", "left hand", "right hand", "both fists", "feet"],
}
ACTIVE = ("queued", "running", "cancelling")


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentConfig:
    dataset: str = "bci"
    epochs: int = 5
    batch_size: int = 8
    lr: float = 5e-4
    seed: int = 42
    database: str | None = None
    name: str = ""
    save_weights: bool = False

    def validate(self):
        if self.dataset not in CLASSES:
            raise ValueError("Choose bci or physionet")
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("Epochs must be a positive integer")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Learning rate must be positive and finite")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be between 0 and 4294967295")
        if not self.database_path.is_file():
            raise FileNotFoundError(f"Database not found: {self.database_path}")

    @property
    def database_path(self):
        return Path(self.database).resolve() if self.database else ROOT / "datasets" / f"{self.dataset}_trials.sqlite"


@contextmanager
def history_connection():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        with connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, finished_at TEXT,
                    status TEXT NOT NULL, config_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL DEFAULT '{}', result_json TEXT,
                    error TEXT, pid INTEGER, process_created REAL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS single_active_run
                    ON runs((1)) WHERE status IN ('queued','running','cancelling');
                CREATE TABLE IF NOT EXISTS epochs (
                    run_id TEXT NOT NULL REFERENCES runs(id), epoch INTEGER NOT NULL,
                    loss REAL NOT NULL, accuracy REAL NOT NULL,
                    PRIMARY KEY(run_id, epoch)
                );
            """)
            yield connection
    finally:
        connection.close()


def list_runs():
    with history_connection() as connection:
        rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_run(run_id):
    with history_connection() as connection:
        row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_epochs(run_id):
    with history_connection() as connection:
        return [dict(row) for row in connection.execute(
            "SELECT epoch, loss, accuracy FROM epochs WHERE run_id=? ORDER BY epoch", (run_id,))]


def update_run(run_id, **fields):
    allowed = {"status", "finished_at", "progress_json", "result_json", "error", "pid", "process_created"}
    if not fields or not fields.keys() <= allowed:
        raise ValueError("Invalid experiment update")
    with history_connection() as connection:
        connection.execute("UPDATE runs SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
                           [*fields.values(), run_id])


def recover_interrupted_runs():
    """Detect dead workers without confusing a recycled PID with our process."""
    for run in list_runs():
        if run["status"] not in ACTIVE:
            continue
        if run["pid"] is None:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(run["created_at"])).total_seconds()
            if age < 60:  # Allow the launcher time to attach the worker PID.
                continue
            alive = False
        else:
            try:
                process = psutil.Process(run["pid"])
                alive = process.is_running() and abs(process.create_time() - run["process_created"]) < 0.01
            except psutil.AccessDenied:
                continue
            except psutil.NoSuchProcess:
                alive = False
        if not alive:
            # Re-read status atomically: the worker may just have finished.
            with history_connection() as connection:
                connection.execute(
                    "UPDATE runs SET status='failed', finished_at=?, error=? "
                    "WHERE id=? AND status IN ('queued','running','cancelling')",
                    (now(), "Training process stopped before completing. See worker.log.", run["id"]))


def create_run(config):
    config.validate()
    recover_interrupted_runs()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    values = asdict(config)
    values["database"] = str(config.database_path)
    try:
        with history_connection() as connection:
            connection.execute("INSERT INTO runs(id,created_at,status,config_json) VALUES(?,?,'queued',?)",
                               (run_id, now(), json.dumps(values)))
    except sqlite3.IntegrityError as exc:
        raise RuntimeError("An experiment is already active. Wait for it or cancel it first.") from exc
    (RUNS_DIR / run_id).mkdir(parents=True, exist_ok=True)
    return run_id


def start_run(config):
    run_id = create_run(config)
    try:
        with (RUNS_DIR / run_id / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", run_id],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        try:
            created = psutil.Process(process.pid).create_time()
            update_run(run_id, pid=process.pid, process_created=created)
        except psutil.NoSuchProcess:
            pass  # A very early import failure is recovered by the history view.
    except Exception as exc:
        update_run(run_id, status="failed", finished_at=now(), error=str(exc))
        raise
    return run_id


def cancel_run(run_id):
    run = get_run(run_id)
    if run and run["status"] in ACTIVE:
        (RUNS_DIR / run_id / "cancel.requested").touch()
        with history_connection() as connection:
            connection.execute("UPDATE runs SET status='cancelling' WHERE id=? "
                               "AND status IN ('queued','running')", (run_id,))


class Cancelled(Exception):
    pass


def run_experiment(config, output_dir=None, on_progress=None, is_cancelled=None):
    """Train the existing full-trial model and evaluate once on an 80/20 split."""
    import numpy as np
    import sklearn
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    from bci_database import load_bci_trials
    from physionet_database import load_physionet_trials
    from model import EEGTransformerModel

    config.validate()
    output_dir = Path(output_dir) if output_dir else None
    started = time.perf_counter()

    def check_cancel():
        if is_cancelled and is_cancelled():
            raise Cancelled()

    def report(**event):
        check_cancel()
        if on_progress:
            on_progress(event)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report(stage="loading", message="Loading cached trials")
    database = config.database_path
    # Record the source manifest without reading several GiB of signal BLOBs.
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
        preprocessing = source.execute("SELECT config_json FROM preprocessing ORDER BY id").fetchall()
        sources = source.execute("SELECT source_sha256 FROM recordings ORDER BY id").fetchall()
    identity = {"path": str(database), "size_bytes": database.stat().st_size,
                "mtime_ns": database.stat().st_mtime_ns,
                "source_manifest_sha256": hashlib.sha256(json.dumps(sources).encode()).hexdigest(),
                "preprocessing": [json.loads(row[0]) for row in preprocessing]}
    loader = load_bci_trials if config.dataset == "bci" else load_physionet_trials
    kwargs = {} if config.dataset == "bci" else {"task_type": "imagery"}
    X, y, channels, metadata = loader(database, **kwargs)
    check_cancel()
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2,
                                          random_state=config.seed, stratify=y)
    train_ids = np.array([metadata[i]["trial_id"] for i in train_idx])
    test_ids = np.array([metadata[i]["trial_id"] for i in test_idx])
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    del X, metadata
    report(stage="normalizing", message="Computing training-only normalization")
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(X_train.std(axis=(0, 2), keepdims=True), np.finfo(np.float32).eps)
    X_train -= mean
    X_train /= std
    X_test -= mean
    X_test /= std
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Inputs contain nonfinite values after normalization")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "split_and_normalization.npz", train_trial_ids=train_ids,
                 test_trial_ids=test_ids, channel_mean=mean, channel_std=std)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=config.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed))
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                             batch_size=config.batch_size, shuffle=False)
    classes = CLASSES[config.dataset]
    architecture = dict(input_channels=len(channels), seq_len=X_train.shape[-1],
                        num_classes=len(classes), d_model=64, nhead=4,
                        num_encoder_layers=1, embedding_type="sinusoidal", conv_block_type="multi")
    model = EEGTransformerModel(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    progress_base = dict(train_count=len(y_train), test_count=len(y_test), device=str(device))
    print(f"Dataset: {config.dataset}; device: {device}; train/test: {len(y_train)}/{len(y_test)}", flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for batch, (xb, yb) in enumerate(train_loader, 1):
            check_cancel()
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite; try a lower learning rate")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
            if batch == 1 or batch % 20 == 0 or batch == len(train_loader):
                report(stage="training", epoch=epoch, batch=batch, batches=len(train_loader),
                       fraction=((epoch - 1) + batch / len(train_loader)) / config.epochs,
                       **progress_base)
        metrics = dict(epoch=epoch, loss=loss_sum / total, accuracy=correct / total)
        report(stage="epoch_complete", **metrics, fraction=epoch / config.epochs, **progress_base)
        print(f"Epoch {epoch}/{config.epochs}: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}", flush=True)
    model.eval()
    loss_sum, predictions = 0.0, []
    report(stage="testing", fraction=1.0, message="Evaluating the held-out 20%", **progress_base)
    with torch.no_grad():
        for xb, yb in test_loader:
            check_cancel()
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_sum += criterion(logits, yb).item() * len(yb)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    result = dict(test_loss=loss_sum / len(y_test), accuracy=float(accuracy_score(y_test, predictions)),
                  balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
                  confusion_matrix=confusion_matrix(y_test, predictions, labels=list(range(len(classes)))).tolist(),
                  class_names=classes, train_count=len(y_train), test_count=len(y_test),
                  train_class_counts=np.bincount(y_train, minlength=len(classes)).tolist(),
                  test_class_counts=np.bincount(y_test, minlength=len(classes)).tolist(),
                  elapsed_seconds=time.perf_counter() - started, device=str(device),
                  gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  sklearn_version=sklearn.__version__,
                  optimizer={"name": "AdamW", "lr": config.lr, "weight_decay": 0.01},
                  normalization="per-channel training mean/std over trials and time",
                  code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("experiments.py", "model.py", "positional_embedding.py")},
                  database_identity=identity, model=architecture, channel_names=channels,
                  protocol="stratified 80/20 trial split; training-only channel z-score; fixed epochs")
    check_cancel()
    if output_dir:
        np.savez(output_dir / "predictions.npz", trial_ids=test_ids,
                 targets=y_test, predictions=np.asarray(predictions))
        if config.save_weights:
            torch.save(model.state_dict(), output_dir / "model.pt")
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test accuracy={result['accuracy']:.4f}, balanced accuracy={result['balanced_accuracy']:.4f}", flush=True)
    return result


def execute_run(run_id):
    run = get_run(run_id)
    if not run or run["status"] not in ACTIVE:
        raise ValueError("Run is not available for execution")
    config = ExperimentConfig(**json.loads(run["config_json"]))
    folder = RUNS_DIR / run_id
    (folder / "config.json").write_text(run["config_json"], encoding="utf-8")
    process = psutil.Process()
    with history_connection() as connection:
        connection.execute("UPDATE runs SET status=CASE WHEN status='cancelling' THEN status ELSE 'running' END, "
                           "pid=?,process_created=? WHERE id=?", (process.pid, process.create_time(), run_id))

    def progress(event):
        with history_connection() as connection:
            connection.execute("UPDATE runs SET progress_json=? WHERE id=?", (json.dumps(event), run_id))
            if event["stage"] == "epoch_complete":
                connection.execute("INSERT INTO epochs VALUES(?,?,?,?)",
                                   (run_id, event["epoch"], event["loss"], event["accuracy"]))
    try:
        result = run_experiment(config, folder, progress, lambda: (folder / "cancel.requested").exists())
        update_run(run_id, status="completed", finished_at=now(), result_json=json.dumps(result))
        return result
    except Cancelled:
        update_run(run_id, status="cancelled", finished_at=now())
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        update_run(run_id, status="failed", finished_at=now(), error=error)
        raise


def cli():
    parser = argparse.ArgumentParser(description="Train the EEG Transformer and save experiment results.")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=list(CLASSES), default="bci")
    parser.add_argument("--database")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="")
    parser.add_argument("--save-weights", action="store_true")
    args = vars(parser.parse_args())
    worker = args.pop("worker")
    if worker:
        execute_run(worker)
    else:
        config = ExperimentConfig(**args)
        run_id = create_run(config)
        print(f"Saved run: {run_id}", flush=True)
        execute_run(run_id)


if __name__ == "__main__":
    cli()
