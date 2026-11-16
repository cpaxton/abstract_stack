"""Locate and fetch the upstream MABEL model without installing its hardware stack."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

MABEL_REPOSITORY = "https://github.com/robotmabel/open_mabel.git"
MABEL_REVISION = "d0181f04025c06e1790b7ab41fe44d602cc2ee5b"
MODEL_RELATIVE_PATH = Path("simulation/mabel_mujoco/models/mabel_full.xml")


def default_mabel_root() -> Path:
    """Use an explicit checkout, a source checkout's third_party, or the user cache."""
    if root := os.environ.get("MABEL_ROOT"):
        return Path(root).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root / "third_party" / "open_mabel"
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "abstract-stack" / "open_mabel"


def mabel_model_path(root: str | Path | None = None) -> Path:
    checkout = Path(root).expanduser().resolve() if root is not None else default_mabel_root()
    model = checkout / MODEL_RELATIVE_PATH
    if not model.is_file():
        raise FileNotFoundError(
            f"MABEL model not found at {model}. Run `abstract-stack fetch-mabel` "
            "or set MABEL_ROOT to an existing open_mabel checkout."
        )
    return model


def fetch_mabel(destination: str | Path | None = None) -> Path:
    """Fetch pinned model and mesh files. Never modify an existing checkout."""
    checkout = Path(destination).expanduser().resolve() if destination else default_mabel_root()
    if checkout.exists():
        model = mabel_model_path(checkout)
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != MABEL_REVISION:
            raise RuntimeError(
                f"Existing MABEL checkout is at {revision}; expected {MABEL_REVISION}. "
                "Use --destination for a new checkout, or pass this one explicitly "
                "with --mabel-root."
            )
        return model
    checkout.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")

    # Publish the checkout only after a successful fetch. Failed downloads can
    # then be retried without deleting or resetting a user's existing directory.
    with tempfile.TemporaryDirectory(prefix=".mabel-fetch-", dir=checkout.parent) as staging:
        staged_checkout = Path(staging) / "open_mabel"

        def git(*args: str) -> None:
            subprocess.run(["git", "-C", str(staged_checkout), *args], env=env, check=True)

        subprocess.run(["git", "init", str(staged_checkout)], env=env, check=True)
        git("remote", "add", "origin", MABEL_REPOSITORY)
        git("sparse-checkout", "init", "--cone")
        git(
            "sparse-checkout",
            "set",
            "simulation/mabel_mujoco/models",
            "simulation/mabel_mujoco/assets",
        )
        git("fetch", "--depth", "1", "--filter=blob:none", "origin", MABEL_REVISION)
        git("checkout", "--detach", "FETCH_HEAD")
        mabel_model_path(staged_checkout)
        staged_checkout.rename(checkout)
    return mabel_model_path(checkout)
