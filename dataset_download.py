"""Download only the explicitly shared SQLite dataset files."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile

FILE_IDS = {
    "bci": "1fprrIocn_bvNifjrBzEBZwq8dWr36hiz",
    "physionet": "13zEKZPcTOpc4tABZKjTCetiY0grwCbxz",
}


def download_dataset(dataset, folder=None):
    """Return (path, downloaded); existing files are never overwritten."""
    if dataset not in FILE_IDS:
        raise ValueError("No download link configured for this dataset")
    folder = Path(folder) if folder else Path(__file__).resolve().parent / "datasets"
    target = folder / f"{dataset}_trials.sqlite"
    if target.exists():
        return target, False
    import gdown
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset_download_", dir=folder) as temporary:
        partial = Path(temporary) / "download.sqlite"
        result = gdown.download(id=FILE_IDS[dataset], output=str(partial), quiet=True,
                                use_cookies=False)
        if not result:
            raise RuntimeError("Google Drive download failed. Check file sharing and download quota.")
        with partial.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise ValueError("The shared file is not a SQLite database")
        with closing(sqlite3.connect(partial.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("SELECT source_sha256 FROM recordings LIMIT 1")
            connection.execute("SELECT config_json FROM preprocessing LIMIT 1")
            connection.execute("SELECT signal_npy, class_name, recording_id FROM trials LIMIT 1")
            if dataset == "physionet":
                connection.execute("SELECT task_type FROM recordings LIMIT 1")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Downloaded database failed integrity validation")
        # Hard-link publication is atomic and refuses to overwrite an existing file.
        try:
            target.hardlink_to(partial)
        except FileExistsError:
            return target, False
    return target, True
