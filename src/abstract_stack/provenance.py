"""Identify the actual Python control/evaluation sources independently of model assets."""

import hashlib
import subprocess
from pathlib import Path


def policy_provenance():
    package = Path(__file__).resolve().parent
    hashes = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
    revision, dirty = None, None
    source_root = package.parents[1]
    if (source_root / "pyproject.toml").is_file():
        try:
            top = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=False,
            )
            if top.returncode != 0 or Path(top.stdout.strip()).resolve() != source_root:
                return {"revision": None, "dirty_sources": None, "python_sha256": hashes}
            result = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return {"revision": None, "dirty_sources": None, "python_sha256": hashes}
        if result.returncode == 0:
            revision = result.stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(source_root), "status", "--porcelain", "--", "src"],
                capture_output=True,
                text=True,
                check=False,
            )
            dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return {"revision": revision, "dirty_sources": dirty, "python_sha256": hashes}
