"""Shared helpers for the R04 review reproduction scripts.

All scripts only READ the GitHub record (git clone at the review commit) and the
local R04 run folder. They write one JSON file each under ./outputs and never
train, predict, open a camera, or modify an input.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]                       # ...\new-chat\work
ARCHIVE = WORK / "github_private_archive"    # git clone, review commit below
RECORD = ARCHIVE / "R04_PAUSED"              # GitHub pause record
R04 = WORK / "r04"                           # local run folder (weights/manifests; not in GitHub)
OUT = HERE / "outputs"
REVIEW_COMMIT = "17f72fdd9143bca6645f9b342414b0c5a2c1ed6d"
VENV_SITE = Path(r"C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Lib\site-packages")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ARCHIVE), *args], capture_output=True, text=True, check=True).stdout.strip()


class Inputs:
    """Records every file a script read, with SHA-256 and size."""

    def __init__(self):
        self.items = {}

    def add(self, path):
        path = Path(path)
        key = str(path.resolve())
        if key not in self.items:
            self.items[key] = {"path": key, "sha256": sha256(path), "bytes": path.stat().st_size}
        return path

    def json(self, path):
        return json.loads(self.add(path).read_text(encoding="utf-8-sig"))


def write_output(name, definitions, results, inputs):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"script": name, "generated_utc": datetime.now(timezone.utc).isoformat(),
               "review_commit": REVIEW_COMMIT, "inputs_modified": False,
               "definitions": definitions, "results": results,
               "inputs": sorted(inputs.items.values(), key=lambda item: item["path"])}
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"script": name, "output": str(path), "inputs": len(inputs.items)}, ensure_ascii=False))
    return payload
