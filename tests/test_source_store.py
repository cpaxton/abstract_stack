"""Exact dirty sources are shared once, portable offline, and fail closed."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from abstract_stack.source_store import freeze_source, resolve_source_reference


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "repo"
    (root / "src/example").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "src/example/__init__.py").write_text("VALUE = 7\n")
    (root / "src/example/extra.py").write_text("OTHER = 8\n")
    (root / "src/example/settings.json").write_text('{"scale": 3}\n')
    (root / "scripts/run.py").write_text(
        "import json\nfrom pathlib import Path\nimport example\n"
        "config = json.loads((Path(example.__file__).parent/'settings.json').read_text())\n"
        "print(example.VALUE * config['scale'])\n"
    )
    (root / "pyproject.toml").write_text('[project]\nname="example"\nversion="0.0.0"\n')
    (root / "uv.lock").write_text("version = 1\n")
    for args in (
        ["init", "-q"],
        ["config", "user.name", "Source Test"],
        ["config", "user.email", "source-test@example.invalid"],
        ["add", "."],
        ["commit", "-qm", "Source fixture"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    paths = ["src/example", "scripts/run.py", "pyproject.toml", "uv.lock"]
    return root, paths, tmp_path / "bundles", tmp_path / "runtime"


def capture(source_tree, **kwargs):
    root, paths, store, cache = source_tree
    return freeze_source(root, paths=paths, store=store, cache=cache, **kwargs)


def test_identical_captures_share_archive_and_runtime_without_per_run_python(source_tree):
    root, _, store, cache = source_tree
    first, second = capture(source_tree), capture(source_tree)
    assert first.content_id == second.content_id and first.archive == second.archive
    assert first.runtime_root == second.runtime_root
    assert list(store.glob("*.zip")) == [first.archive]
    assert len(list(cache.iterdir())) == 1
    assert not first.git["dirty_sources"] and len(first.git["revision"]) == 40
    assert len(first.git["tree"]) == 40
    run = root / "outputs/run"
    first.write_reference(run / "source_ref.json")
    assert not list(run.rglob("*.py"))
    assert json.loads((run / "source_ref.json").read_text())["git"] == first.git
    with pytest.raises(FileExistsError):
        first.write_reference(run / "source_ref.json")


def test_dirty_and_untracked_selected_code_is_exact_and_unrelated_secrets_are_excluded(source_tree):
    root, _, _, _ = source_tree
    clean = capture(source_tree)
    (root / "src/example/__init__.py").write_text("VALUE = 11\n")
    (root / "src/example/new_policy.py").write_text("NEW = True\n")
    (root / "src/example/.env").write_text("SECRET=do-not-copy-this\n")
    (root / "private.env").write_text("PRIVATE=unrelated\n")
    (root / "outputs").mkdir()
    (root / "outputs/private.py").write_text("SECRET = 'outside source scope'\n")
    dirty = capture(source_tree)
    assert dirty.content_id != clean.content_id and dirty.git["dirty_sources"]
    assert any("new_policy.py" in p for p in dirty.git["status"])
    assert not any(".env" in p or "outputs/" in p for p in dirty.git["status"])
    with zipfile.ZipFile(dirty.archive) as archive:
        assert archive.read("src/example/__init__.py") == b"VALUE = 11\n"
        assert archive.read("src/example/new_policy.py") == b"NEW = True\n"
        assert not any(".env" in name or "outputs/" in name for name in archive.namelist())


def test_deleted_tracked_file_changes_identity_and_is_recorded_in_git_metadata(source_tree):
    root, _, _, _ = source_tree
    clean = capture(source_tree)
    (root / "src/example/extra.py").unlink()
    removed = capture(source_tree)
    assert removed.content_id != clean.content_id
    assert removed.git["dirty_sources"]
    assert removed.git["deleted_files"] == ["src/example/extra.py"]
    assert "src/example/extra.py" not in removed.files


def test_same_source_bytes_share_bundle_across_unrelated_git_commits(source_tree):
    root, _, _, _ = source_tree
    before = capture(source_tree)
    (root / "README.md").write_text("Unrelated documentation\n")
    subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "Docs only"], check=True)
    after = capture(source_tree)
    assert before.git["revision"] != after.git["revision"]
    assert before.content_id == after.content_id and before.archive == after.archive
    assert before.runtime_root == after.runtime_root and not after.git["dirty_sources"]


def test_ignored_but_selected_source_is_recorded_as_dirty(source_tree):
    root, _, _, _ = source_tree
    (root / ".gitignore").write_text("src/example/ignored_policy.py\n")
    (root / "src/example/ignored_policy.py").write_text("IGNORED = True\n")
    frozen = capture(source_tree)
    assert frozen.git["dirty_sources"]
    assert any("ignored_policy.py" in p for p in frozen.git["status"])
    assert "src/example/ignored_policy.py" in frozen.files


def test_sources_changing_during_freeze_fail_before_publishing_a_bundle(source_tree, monkeypatch):
    from abstract_stack import source_store

    root, _, store, _ = source_tree
    original = source_store._collect
    count = 0

    def changing(*args):
        nonlocal count
        result = original(*args)
        count += 1
        if count == 1:
            (root / "src/example/__init__.py").write_text("VALUE = 42\n")
        return result

    monkeypatch.setattr(source_store, "_collect", changing)
    with pytest.raises(ValueError, match="changed during capture"):
        capture(source_tree)
    assert not store.exists()


def test_portable_bundle_runs_offline_without_original_git_or_shared_archive(source_tree, tmp_path):
    root, _, _, _ = source_tree
    (root / "src/example/__init__.py").write_text("VALUE = 13\n")
    frozen = capture(source_tree, materialize=False)
    assert frozen.runtime_root is None
    reference = tmp_path / "export/source_ref.json"
    frozen.write_reference(reference, portable=True)
    frozen.archive.unlink()
    root.rename(tmp_path / "unavailable-checkout")
    restored = resolve_source_reference(reference, cache=tmp_path / "offline-runtime")
    result = subprocess.run(
        [sys.executable, str(restored.runtime_root / "scripts/run.py")],
        env=dict(
            os.environ, PYTHONPATH=str(restored.runtime_root / "src"), PYTHONDONTWRITEBYTECODE="1"
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "39"
    assert restored.git["dirty_sources"] and not (root / ".git").exists()


def test_missing_shared_archive_never_falls_back_to_the_live_checkout(source_tree, tmp_path):
    frozen = capture(source_tree, materialize=False)
    reference = tmp_path / "run/source_ref.json"
    frozen.write_reference(reference)
    frozen.archive.unlink()
    with pytest.raises(ValueError, match="Source archive missing"):
        resolve_source_reference(reference, cache=tmp_path / "runtime")


def test_modified_runtime_is_rejected_and_preserved_instead_of_silently_refreshed(source_tree):
    frozen = capture(source_tree)
    source = frozen.runtime_root / "src/example/__init__.py"
    source.chmod(0o644)
    source.write_text("VALUE = 999\n")
    with pytest.raises(ValueError, match="source file is writable"):
        capture(source_tree)
    assert source.read_text() == "VALUE = 999\n"


def test_added_runtime_module_is_rejected_even_when_original_files_are_unchanged(source_tree):
    frozen = capture(source_tree)
    package = frozen.runtime_root / "src/example"
    package.chmod(0o755)
    (package / "unrecorded.py").write_text("EXTRA = True\n")
    with pytest.raises(ValueError, match="source directory is writable"):
        capture(source_tree)


def test_cached_python_bytecode_is_rejected_before_a_new_worker_can_import_it(source_tree):
    frozen = capture(source_tree)
    package = frozen.runtime_root / "src/example"
    package.chmod(0o755)
    bytecode = package / "__pycache__"
    bytecode.mkdir()
    (bytecode / "__init__.pyc").write_bytes(b"unrecorded cached policy")
    package.chmod(0o555)
    with pytest.raises(ValueError, match="Unexpected runtime bytecode"):
        capture(source_tree)


def test_corrupt_archive_is_rejected_before_materialization(source_tree, tmp_path):
    frozen = capture(source_tree, materialize=False)
    reference = tmp_path / "export/source_ref.json"
    frozen.write_reference(reference, portable=True)
    with (reference.parent / "source_bundle.zip").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_source_reference(reference, cache=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", "."])
def test_source_scopes_cannot_escape_or_capture_the_whole_workspace(source_tree, name):
    root, _, store, cache = source_tree
    with pytest.raises(ValueError, match="relative paths"):
        freeze_source(root, paths=[name], store=store, cache=cache)


def test_source_symlinks_are_rejected(source_tree):
    root, _, _, _ = source_tree
    (root / "src/example/link.py").symlink_to(root / "private.env")
    (root / "private.env").write_text("private\n")
    with pytest.raises(ValueError, match="escapes"):
        capture(source_tree)


def test_archive_path_traversal_cannot_write_outside_cache(tmp_path):
    payload = b"VALUE = 1\n"
    files = {
        "../outside.py": dict(
            sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), executable=False
        )
    }
    identity = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    archive = tmp_path / "source_bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "_source_manifest.json",
            json.dumps(dict(schema_version=1, content_id=identity, files=files)),
        )
        bundle.writestr("../outside.py", payload)
    reference = tmp_path / "source_ref.json"
    reference.write_text(
        json.dumps(
            dict(
                schema_version=1,
                content_id=identity,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                archive="source_bundle.zip",
                portable=True,
                git={},
                files=files,
            )
        )
    )
    with pytest.raises(ValueError, match="relative paths"):
        resolve_source_reference(reference, cache=tmp_path / "cache")
    assert not (tmp_path / "outside.py").exists()


def test_source_reference_inspection_needs_no_native_simulator_import(source_tree, tmp_path):
    frozen = capture(source_tree, materialize=False)
    reference = tmp_path / "export/source_ref.json"
    frozen.write_reference(reference, portable=True)
    code = (
        "import sys;from abstract_stack.source_store import resolve_source_reference;"
        "resolve_source_reference(sys.argv[1],materialize=False);"
        "assert 'mujoco' not in sys.modules and 'numpy' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(reference)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_frozen_policy_inherits_git_only_after_matching_actual_imported_sources(
    source_tree, tmp_path, monkeypatch
):
    from abstract_stack import provenance

    frozen = capture(source_tree)
    reference = tmp_path / "run/source_ref.json"
    frozen.write_reference(reference)
    source = frozen.runtime_root / "src/example/__init__.py"
    monkeypatch.setattr(provenance, "__file__", str(source))
    monkeypatch.setenv("ABSTRACT_STACK_SOURCE_REFERENCE", str(reference))
    result = provenance.policy_provenance()
    assert result["revision"] == frozen.git["revision"]
    assert result["dirty_sources"] is False and result["source_bundle_id"] == frozen.content_id
    source.chmod(0o644)
    source.write_text("VALUE = 999\n")
    with pytest.raises(ValueError, match="differs from the imported"):
        provenance.policy_provenance()


def test_frozen_child_environment_retains_scoped_git_without_importing_native_simulator(
    source_tree, tmp_path
):
    from abstract_stack import provenance, source_store

    root, _, store, cache = source_tree
    package = root / "src/abstract_stack"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for module in (provenance, source_store):
        source = Path(module.__file__)
        (package / source.name).write_bytes(source.read_bytes())
    frozen = freeze_source(
        root, paths=["src/abstract_stack", "pyproject.toml", "uv.lock"], store=store, cache=cache
    )
    reference = tmp_path / "run/source_ref.json"
    frozen.write_reference(reference)
    command = (
        "import json,sys;from abstract_stack.provenance import policy_provenance;"
        "print(json.dumps(policy_provenance()));"
        "assert 'mujoco' not in sys.modules and 'numpy' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env=dict(
            os.environ,
            PYTHONPATH=str(frozen.runtime_root / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            ABSTRACT_STACK_SOURCE_REFERENCE=str(reference),
        ),
    )
    recorded = json.loads(result.stdout)
    assert recorded["revision"] == frozen.git["revision"]
    assert recorded["dirty_sources"] and recorded["source_bundle_id"] == frozen.content_id


def test_nested_capture_inherits_git_only_for_verified_parent_source_subset(
    source_tree, tmp_path, monkeypatch
):
    parent = capture(source_tree)
    reference = tmp_path / "campaign/source_ref.json"
    parent.write_reference(reference)
    monkeypatch.setenv("ABSTRACT_STACK_SOURCE_REFERENCE", str(reference))
    nested = freeze_source(
        parent.runtime_root,
        paths=["src/example", "uv.lock"],
        store=parent.archive.parent,
        cache=tmp_path / "nested-runtime",
    )
    assert nested.git["revision"] == parent.git["revision"]
    assert nested.git["inherited_source_bundle_id"] == parent.content_id
    assert nested.git["dirty_sources"] == parent.git["dirty_sources"]
    assert nested.content_id != parent.content_id


def test_nested_git_inheritance_never_labels_changed_files_with_parent_revision(
    source_tree, tmp_path, monkeypatch
):
    parent = capture(source_tree)
    reference = tmp_path / "campaign/source_ref.json"
    parent.write_reference(reference)
    standalone = tmp_path / "independent"
    (standalone / "src/example").mkdir(parents=True)
    (standalone / "src/example/__init__.py").write_text("VALUE = 999\n")
    monkeypatch.setenv("ABSTRACT_STACK_SOURCE_REFERENCE", str(reference))
    independent = freeze_source(
        standalone, paths=["src/example"], store=parent.archive.parent, materialize=False
    )
    assert independent.git["revision"] is None and independent.git["dirty_sources"] is None
