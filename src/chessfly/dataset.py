"""Public MaleCNS v1.0 dataset inventory and atomic downloader."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import certifi


BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"


@dataclass(frozen=True)
class DatasetFile:
    key: str
    filename: str
    url: str
    approximate_bytes: int
    purpose: str


DATASET_FILES = (
    DatasetFile(
        "annotations",
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        f"{BASE_URL}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        13_000_000,
        "curated neuron classes, types, sides, and related annotations",
    ),
    DatasetFile(
        "transmitters",
        "body-neurotransmitters-male-cns-v1.0.feather",
        f"{BASE_URL}/body-neurotransmitters-male-cns-v1.0.feather",
        42_000_000,
        "aggregate neurotransmitter predictions per neuron",
    ),
    DatasetFile(
        "weights",
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        f"{BASE_URL}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        1_100_000_000,
        "complete segment-to-segment connection strengths",
    ),
)


def select_files(keys: Iterable[str]) -> tuple[DatasetFile, ...]:
    requested = tuple(keys)
    if "all" in requested:
        return DATASET_FILES
    by_key = {item.key: item for item in DATASET_FILES}
    unknown = set(requested) - set(by_key)
    if unknown:
        raise ValueError(f"unknown MaleCNS files: {sorted(unknown)}")
    return tuple(by_key[key] for key in requested)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(item: DatasetFile, raw_dir: Path) -> dict[str, object]:
    """Download a source file atomically and return its local provenance."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / item.filename
    if target.exists():
        return _local_record(item, target, "existing")

    temp_path: Optional[Path] = None
    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=raw_dir, prefix=f".{item.key}-", delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(item.url, context=tls_context) as response:
                with temp_path.open("wb") as temporary:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        temporary.write(chunk)
        except OSError as urllib_error:
            curl = shutil.which("curl")
            if not curl:
                raise RuntimeError(
                    "Python HTTPS failed and curl is unavailable"
                ) from urllib_error
            try:
                subprocess.run(
                    [
                        curl,
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--output",
                        str(temp_path),
                        item.url,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as curl_error:
                raise RuntimeError(
                    f"both Python HTTPS and curl failed for {item.url}"
                ) from curl_error
        os.replace(temp_path, target)
        temp_path = None
        return _local_record(item, target, "downloaded")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_manifest(records: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, object]] = {}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        existing = {item["key"]: item for item in previous.get("files", [])}
    for record in records:
        existing[str(record["key"])] = record
    payload = {
        "schema_version": 1,
        "dataset": "male-cns:v1.0",
        "license": "CC-BY-4.0",
        "source": "https://male-cns.janelia.org/download/",
        "files": [existing[key] for key in sorted(existing)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inventory() -> list[dict[str, object]]:
    return [asdict(item) for item in DATASET_FILES]


def _local_record(item: DatasetFile, path: Path, state: str) -> dict[str, object]:
    return {
        "key": item.key,
        "filename": item.filename,
        "url": item.url,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "state": state,
    }
