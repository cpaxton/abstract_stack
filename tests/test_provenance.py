"""A source archive must not inherit the identity of an enclosing checkout."""

import os
import shutil
import subprocess

import pytest

from abstract_stack import provenance


@pytest.mark.parametrize("archived", [False, True])
def test_source_revision_is_only_attributed_to_its_own_checkout(tmp_path, monkeypatch, archived):
    if shutil.which("git") is None:
        pytest.skip("Git not installed")
    for key in list(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, input=None):
        return subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                *args,
            ],
            input=input,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    git("init")
    tree = git("mktree", input="")
    head = git("commit-tree", tree, "-m", "Empty provenance fixture")
    git("update-ref", "HEAD", head)
    (repo / ".gitignore").write_text("outputs/\n")
    root = repo / "outputs/archive" if archived else repo
    package = root / "src/abstract_stack"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    source = package / "provenance.py"
    source.write_text("# Independent source bytes\n")
    monkeypatch.setattr(provenance, "__file__", str(source))
    result = provenance.policy_provenance()
    assert len(result["python_sha256"]["provenance.py"]) == 64
    if archived:
        assert result["revision"] is None
        assert result["dirty_sources"] is None
    else:
        assert result["revision"] == head
        assert result["dirty_sources"] is True
