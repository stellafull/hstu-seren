from __future__ import annotations

import gzip
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "tmp"

DatasetConfig = Mapping[str, Any]


DATASETS: dict[str, DatasetConfig] = {
    "serendipity-2018": {
        "url": "https://files.grouplens.org/datasets/serendipity-sac2018/serendipity-sac2018.zip",
        "type": "zip",
    },
    "amzn_books_2015": {
        "urls": [
            (
                "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz",
                "meta_Books.json.gz",
            ),
            (
                "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
                "ratings_Books.csv",
            ),
            (
                "https://raw.githubusercontent.com/zhefu2/SerenLens/master/Dataset/SerenLens_Books.csv",
                "SerenLens_Books.csv",
            ),
        ],
        "gunzip": [("meta_Books.json.gz", "meta_Books.json")],
        "lfs_files": [
            {
                "remote": "Dataset/SerenLens_Books.csv",
                "local": "SerenLens_Books.csv",
            }
        ],
    },
    "amzn_mv_2015": {
        "urls": [
            (
                "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz",
                "meta_Movies_and_TV.json.gz",
            ),
            (
                "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv",
                "ratings_Movies_and_TV.csv",
            ),
            (
                "https://raw.githubusercontent.com/zhefu2/SerenLens/master/Dataset/SerenLens_Movies.csv",
                "SerenLens_Movies.csv",
            ),
        ],
        "gunzip": [("meta_Movies_and_TV.json.gz", "meta_Movies_and_TV.json")],
        "lfs_files": [
            {
                "remote": "Dataset/SerenLens_Movies.csv",
                "local": "SerenLens_Movies.csv",
            }
        ],
    },
}

LFS_CONFIG = {"owner": "zhefu2", "repo": "SerenLens", "ref": "refs/heads/master"}


def _config_pairs(values: Any) -> Iterator[tuple[str, str]]:
    if not values:
        return
    for src, dst in values:
        yield str(src), str(dst)


def _lfs_entries(config: DatasetConfig) -> Iterator[tuple[str, str]]:
    for entry in config.get("lfs_files", []) or []:
        if isinstance(entry, MappingABC):
            remote = entry.get("remote")
            local = entry.get("local") or remote
        else:
            remote = entry
            local = entry
        if remote and local:
            yield str(remote), str(local)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
            return file_obj.readline().startswith(
                "version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return True


def _download(
    url: str,
    filepath: Path,
    headers: Optional[Mapping[str, str]] = None,
    *,
    overwrite: bool = False,
) -> None:
    """Download a file from URL to *filepath* if missing."""
    if filepath.exists() and not overwrite:
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    req_headers = {"User-Agent": "Mozilla/5.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req) as resp, open(filepath, "wb") as file_obj:
        shutil.copyfileobj(resp, file_obj, length=1024 * 1024)


def _gunzip_file(src: Path, dst: Path) -> None:
    """Decompress gzip file if destination doesn't exist, then delete the .gz file."""
    if not src.exists() or dst.exists():
        return

    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
    
    # Delete the .gz file after successful decompression to save space
    try:
        src.unlink()
        log.info("Deleted compressed file %s after extraction", src)
    except Exception as exc:
        log.warning("Failed to delete compressed file %s: %s", src, exc)


def _resolve_lfs_file(filepath: Path, remote_path: str) -> None:
    """Resolve Git LFS pointer files via GitHub's batch API."""
    if not filepath.exists():
        return

    try:
        with open(filepath, "rb") as file_obj:
            head = file_obj.read(128).decode("utf-8", errors="ignore")
        if not head.startswith("version https://git-lfs.github.com/spec/v1"):
            return
    except Exception:
        return

    oid: Optional[str] = None
    size: Optional[int] = None
    try:
        with open(filepath, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.startswith("oid sha256:"):
                    oid = line.split(":", 1)[1].strip()
                elif line.startswith("size "):
                    try:
                        size = int(line.split()[1])
                    except (IndexError, ValueError):
                        size = None
        if not oid or not size:
            return

        branch = LFS_CONFIG["ref"].rsplit("/", 1)[-1]
        media_url = (
            f"https://media.githubusercontent.com/media/{LFS_CONFIG['owner']}/"
            f"{LFS_CONFIG['repo']}/{branch}/{remote_path}"
        )
        try:
            _download(media_url, filepath, overwrite=True)
            return
        except Exception as exc:
            log.warning(
                "Direct media download failed for %s (remote %s): %s", filepath, remote_path, exc
            )

        payload = {
            "operation": "download",
            "objects": [{"oid": oid, "size": size}],
            "transfers": ["basic"],
            "ref": {"name": LFS_CONFIG["ref"]},
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "Mozilla/5.0",
        }
        batch_url = (
            f"https://github.com/{LFS_CONFIG['owner']}/{LFS_CONFIG['repo']}.git/info/lfs/objects/batch"
        )
        request = urllib.request.Request(
            batch_url, data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
        download_info = result.get("objects", [{}])[0].get("actions", {}).get("download")
        if download_info and "href" in download_info:
            _download(
                download_info["href"],
                filepath,
                headers=download_info.get("header", {}),
                overwrite=True,
            )
    except Exception as exc:
        log.warning("Failed to resolve LFS file %s: %s", filepath, exc)
        return


def _dataset_exists(dataset_path: Path, dataset_config: DatasetConfig) -> bool:
    if not dataset_path.is_dir():
        return False

    dataset_type = dataset_config.get("type")
    if dataset_type == "zip":
        try:
            return any(dataset_path.iterdir())
        except OSError:
            return False

    gunzip_targets = {src: dst for src, dst in _config_pairs(dataset_config.get("gunzip"))}
    for _, filename in _config_pairs(dataset_config.get("urls")):
        target = dataset_path / gunzip_targets.get(filename, filename)
        if not target.exists():
            return False

    for _, local in _lfs_entries(dataset_config):
        pointer_path = dataset_path / local
        if pointer_path.exists() and _is_lfs_pointer(pointer_path):
            return False

    return True


def _download_zip_dataset(dataset_path: Path, url: str) -> None:
    dataset_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "dataset.zip"
        _download(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(dataset_path)


def _download_url_dataset(dataset_path: Path, config: DatasetConfig) -> None:
    dataset_path.mkdir(parents=True, exist_ok=True)
    for url, filename in _config_pairs(config.get("urls")):
        _download(url, dataset_path / filename)
    for remote, local in _lfs_entries(config):
        _resolve_lfs_file(dataset_path / local, remote)
    for src, dst in _config_pairs(config.get("gunzip")):
        _gunzip_file(dataset_path / src, dataset_path / dst)


def download_dataset(name: str, root: str | Path | None = None) -> bool:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    root_path = Path(root) if root is not None else DEFAULT_ROOT
    dataset_path = root_path / name
    config = DATASETS[name]

    if _dataset_exists(dataset_path, config):
        return True

    try:
        if config.get("type") == "zip":
            _download_zip_dataset(dataset_path, config["url"])  # type: ignore[index]
        else:
            _download_url_dataset(dataset_path, config)
    except Exception as exc:
        log.exception("Failed to download dataset %s: %s", name, exc)
        return False

    return _dataset_exists(dataset_path, config)


def download_all(root: str | Path | None = None) -> dict[str, bool]:
    return {name: download_dataset(name, root=root) for name in DATASETS}


__all__ = ["DEFAULT_ROOT", "DATASETS", "download_dataset", "download_all"]
