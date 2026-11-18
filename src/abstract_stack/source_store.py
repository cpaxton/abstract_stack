"""Shared, offline source bundles with scoped Git metadata and exact dirty bytes.

Runs reference a content-addressed ZIP. Runnable trees are verified and shared
under the ignored cache; portable checkpoints can embed the same ZIP. This
module deliberately imports no simulator or third-party packages.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA = 1
MAX_BYTES = 64 * 1024 * 1024
MAX_FILES = 10000
MANIFEST = "_source_manifest.json"
SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml", ".xml", ".txt", ".md", ".sh", ".lock"}
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".ruff_cache", ".pytest_cache"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _relative(value):
    path = PurePosixPath(value)
    if (
        not value
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
        or value == MANIFEST
    ):
        raise ValueError("Source paths must be normalized relative paths")
    return path


def _allowed(path):
    return not path.name.startswith(".") and path.suffix in SOURCE_SUFFIXES


def _collect(root, paths):
    result = {}
    total_bytes = 0
    for name in paths:
        relative = _relative(name)
        source = root.joinpath(*relative.parts)
        if source.is_symlink() or not source.exists():
            raise ValueError(f"Missing or symlinked source: {name}")
        if source.is_file():
            if not _allowed(source):
                raise ValueError(f"Unsupported source file type: {name}")
            candidates = [source]
        else:
            candidates = []
            for parent, dirs, files in os.walk(source):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
                if any((Path(parent) / d).is_symlink() for d in dirs):
                    raise ValueError("Source directory contains a symlink")
                candidates.extend(Path(parent) / f for f in sorted(files) if _allowed(Path(f)))
        for path in candidates:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Source file escapes the selected tree")
            relative_name = path.relative_to(root).as_posix()
            if relative_name in result:
                continue
            with path.open("rb") as stream:
                data = stream.read(MAX_BYTES - total_bytes + 1)
            total_bytes += len(data)
            if total_bytes > MAX_BYTES or len(result) >= MAX_FILES:
                raise ValueError("Source bundle must contain bounded source files")
            result[relative_name] = (data, bool(path.stat().st_mode & 0o111))
    if not result or len(result) > MAX_FILES or sum(len(v[0]) for v in result.values()) > MAX_BYTES:
        raise ValueError("Source bundle must contain bounded source files")
    return result


def _git(root, names, scopes):
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
        )

    empty = dict(revision=None, tree=None, dirty_sources=None, status=None)
    try:
        top = run("rev-parse", "--show-toplevel")
        if top.returncode or Path(top.stdout.strip()).resolve() != root:
            return empty
        head, tree = run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")
        removed = run("diff", "--name-only", "--diff-filter=D", "-z", "HEAD", "--", *scopes)
        deleted = sorted(n for n in removed.stdout.split("\0") if n and _allowed(PurePosixPath(n)))
        status = run(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=matching",
            "--",
            *sorted(set(names) | set(deleted)),
        )
        if head.returncode or tree.returncode or status.returncode:
            return empty
        return dict(
            revision=head.stdout.strip(),
            tree=tree.stdout.strip(),
            dirty_sources=bool(status.stdout),
            status=status.stdout.split("\0")[:-1],
            deleted_files=deleted,
        )
    except (OSError, subprocess.TimeoutExpired):
        return empty


def _file_manifest(files):
    return {
        name: dict(sha256=_sha(data), bytes=len(data), executable=executable)
        for name, (data, executable) in sorted(files.items())
    }


def _read_archive(archive, expected_id=None, expected_sha=None):
    if expected_sha is not None and _sha(archive.read_bytes()) != expected_sha:
        raise ValueError("Source archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if (
            len(infos) > MAX_FILES + 1
            or sum(i.file_size for i in infos) > MAX_BYTES + 4 * 1024 * 1024
            or len({i.filename for i in infos}) != len(infos)
        ):
            raise ValueError("Invalid or oversized source archive")
        manifest = json.loads(bundle.read(MANIFEST))
        files = manifest["files"]
        identity = _sha(_json(files))
        if (
            manifest.get("schema_version") != SCHEMA
            or manifest.get("content_id") != identity
            or expected_id is not None
            and expected_id != identity
            or set(bundle.namelist()) != {MANIFEST, *files}
        ):
            raise ValueError("Source archive identity/manifest mismatch")
        payload = {}
        for name, entry in files.items():
            _relative(name)
            if not _allowed(PurePosixPath(name)):
                raise ValueError("Unsupported file in source archive")
            data = bundle.read(name)
            if len(data) != entry["bytes"] or _sha(data) != entry["sha256"]:
                raise ValueError("Source archive file checksum mismatch")
            payload[name] = (data, entry["executable"])
    return identity, files, payload


def _remove_temporary(path):
    for parent, _dirs, files in os.walk(path):
        Path(parent).chmod(0o755)
        for name in files:
            (Path(parent) / name).chmod(0o644)
    shutil.rmtree(path)


def _verify_runtime(target, files):
    expected_marker = dict(schema_version=SCHEMA, content_id=_sha(_json(files)), files=files)
    if json.loads((target / MANIFEST).read_text()) != expected_marker:
        raise ValueError("Shared runtime source manifest was modified")
    names = set()
    for parent, dirs, leaves in os.walk(target):
        if Path(parent).stat().st_mode & 0o222:
            raise ValueError("Shared runtime source directory is writable")
        if "__pycache__" in dirs:
            raise ValueError("Unexpected runtime bytecode; use PYTHONDONTWRITEBYTECODE=1")
        if any((Path(parent) / d).is_symlink() for d in dirs):
            raise ValueError("Shared runtime source tree contains a symlink")
        for name in leaves:
            path = Path(parent) / name
            if path.stat().st_mode & 0o222:
                raise ValueError("Shared runtime source file is writable")
            relative = path.relative_to(target).as_posix()
            if relative != MANIFEST:
                names.add(relative)
    if names != set(files) or _file_manifest(_collect(target, list(files))) != files:
        raise ValueError("Shared runtime source tree was modified")


def _materialize(archive, cache, identity, archive_sha):
    _, files, payload = _read_archive(archive, identity, archive_sha)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / identity
    if target.exists():
        _verify_runtime(target, files)
        return target
    temporary = Path(tempfile.mkdtemp(prefix=f".{identity[:12]}-", dir=cache))
    try:
        for name, (data, executable) in payload.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o555 if executable else 0o444)
        (temporary / MANIFEST).write_bytes(
            _json(dict(schema_version=SCHEMA, content_id=identity, files=files))
        )
        (temporary / MANIFEST).chmod(0o444)
        for parent, _, _ in os.walk(temporary, topdown=False):
            Path(parent).chmod(0o555)
        try:
            temporary.rename(target)
        except OSError:
            if not target.is_dir():
                raise
            # Another writer may have published the exact same source tree.
            _verify_runtime(target, files)
        return target
    finally:
        if temporary.exists():
            _remove_temporary(temporary)


@dataclass(frozen=True)
class FrozenSource:
    content_id: str
    archive: Path
    archive_sha256: str
    runtime_root: Path | None
    files: dict
    git: dict

    def write_reference(self, destination, *, portable=False):
        """Write run metadata, optionally embedding the compressed archive beside it."""
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive = self.archive
        if portable:
            archive = destination.parent / "source_bundle.zip"
            with archive.open("xb") as target, self.archive.open("rb") as source:
                shutil.copyfileobj(source, target)
        reference = dict(
            schema_version=SCHEMA,
            content_id=self.content_id,
            archive_sha256=self.archive_sha256,
            archive="source_bundle.zip" if portable else str(archive.resolve()),
            portable=portable,
            git=self.git,
            files=self.files,
        )
        with destination.open("x") as stream:
            stream.write(json.dumps(reference, indent=2) + "\n")
        return reference


def default_source_store(root):
    """Prefer the owning repository's shared store, including frozen children."""
    configured = os.environ.get("ABSTRACT_STACK_SOURCE_STORE")
    if configured:
        return Path(configured).resolve()
    root = Path(root).resolve()
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            return parent / "outputs/source_bundles"
    owner = getattr(os, "getuid", lambda: "user")()
    return Path(tempfile.gettempdir()) / f"abstract-stack-{owner}" / "source_bundles"


def freeze_source(root, *, paths, store=None, cache=None, materialize=True):
    """Capture only declared source scopes, then share the verified runnable tree.

    Full exact selected bytes make clean and dirty code recoverable offline,
    independently of later Git garbage collection. Git metadata is supplemental;
    no broad workspace patch, credentials or unrelated outputs are collected.
    """
    root = Path(root).resolve()
    store = Path(store) if store is not None else default_source_store(root)
    cache = Path(cache) if cache is not None else store.parent / "cache/source_runtime"
    paths = tuple(str(PurePosixPath(p)) for p in paths)
    files = _collect(root, paths)
    git = _git(root, sorted(files), paths)
    if files != _collect(root, paths) or git != _git(root, sorted(files), paths):
        raise ValueError("Selected sources changed during capture; start a fresh capture")
    manifest = _file_manifest(files)
    parent_reference = os.environ.get("ABSTRACT_STACK_SOURCE_REFERENCE")
    if git["revision"] is None and parent_reference:
        parent = resolve_source_reference(parent_reference, materialize=False)
        if all(parent.files.get(name) == record for name, record in manifest.items()):
            git = dict(parent.git, inherited_source_bundle_id=parent.content_id)
    identity = _sha(_json(manifest))
    store.mkdir(parents=True, exist_ok=True)
    archive = store / f"{identity}.zip"
    if not archive.exists():
        descriptor, name = tempfile.mkstemp(prefix=".source-", suffix=".zip", dir=store)
        os.close(descriptor)
        temporary = Path(name)
        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as z:
                entries = {
                    MANIFEST: _json(
                        dict(schema_version=SCHEMA, content_id=identity, files=manifest)
                    ),
                    **{key: value[0] for key, value in files.items()},
                }
                for key, data in sorted(entries.items()):
                    info = zipfile.ZipInfo(key, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    executable = key in files and files[key][1]
                    info.external_attr = (0o100555 if executable else 0o100444) << 16
                    z.writestr(info, data)
            try:
                os.link(temporary, archive)
                archive.chmod(0o444)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    archive_sha = _sha(archive.read_bytes())
    runtime = _materialize(archive, cache, identity, archive_sha) if materialize else None
    if runtime is None:
        _read_archive(archive, identity, archive_sha)
    return FrozenSource(
        identity,
        archive.resolve(),
        archive_sha,
        None if runtime is None else runtime.resolve(),
        manifest,
        git,
    )


def resolve_source_reference(reference_path, *, cache=None, materialize=True):
    """Verify a reference/archive and materialize it; missing archives fail closed."""
    reference_path = Path(reference_path)
    reference = json.loads(reference_path.read_text())
    if reference.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported source reference schema")
    archive = Path(reference["archive"])
    if not archive.is_absolute():
        _relative(reference["archive"])
        archive = reference_path.parent / archive
    if not archive.is_file():
        raise ValueError(
            "Source archive missing; export a portable source reference before relocation"
        )
    identity, files, _ = _read_archive(
        archive, reference["content_id"], reference["archive_sha256"]
    )
    if files != reference["files"]:
        raise ValueError("Source reference file manifest differs from archive")
    if cache is None:
        cache = default_source_store(reference_path.parent).parent / "cache/source_runtime"
    runtime = (
        _materialize(archive, Path(cache), identity, reference["archive_sha256"])
        if materialize
        else None
    )
    return FrozenSource(
        identity,
        archive.resolve(),
        reference["archive_sha256"],
        None if runtime is None else runtime.resolve(),
        files,
        reference["git"],
    )
