"""Exercise shell launcher behavior on Linux and macOS without a native viewer."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout with spaces"
    (root / "scripts").mkdir(parents=True)
    for name in ("run-macos.sh", "install-macos.sh"):
        shutil.copy2(Path(__file__).resolve().parents[1] / "scripts" / name, root / "scripts")
    tools = tmp_path / "tools"
    tools.mkdir()
    uname = tools / "uname"
    uname.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_SYSTEM"\n')
    uname.chmod(0o755)
    env = dict(os.environ, PATH=f"{tools}{os.pathsep}{os.defpath}", TEST_SYSTEM="Darwin")
    env.pop("DYLD_FALLBACK_LIBRARY_PATH", None)
    return root, env


def run(checkout, script, *args):
    root, env = checkout
    return subprocess.run(
        [
            "bash",
            "-c",
            # Set DYLD inside Bash: macOS strips it when starting system shells.
            'if [[ -n "${TEST_EXISTING_LIBRARY_PATH:-}" ]]; then '
            'export DYLD_FALLBACK_LIBRARY_PATH="$TEST_EXISTING_LIBRARY_PATH"; fi; source "$@"',
            "test",
            str(root / "scripts" / script),
            *args,
        ],
        cwd=root.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("script", ["run-macos.sh", "install-macos.sh"])
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_without_installation_or_macos(checkout, script, flag):
    _, env = checkout
    env["TEST_SYSTEM"] = "Linux"
    result = run(checkout, script, flag)
    assert result.returncode == 0, result.stderr
    assert f"Usage: scripts/{script}" in result.stdout
    assert "3.10" in result.stdout and "3.12" in result.stdout


def test_launcher_platform_and_missing_environment_errors(checkout):
    _, env = checkout
    env["TEST_SYSTEM"] = "Linux"
    result = run(checkout, "run-macos.sh", "sim")
    assert result.returncode == 1
    assert "requires macOS" in result.stderr
    env["TEST_SYSTEM"] = "Darwin"
    result = run(checkout, "run-macos.sh", "sim")
    assert result.returncode == 1
    assert "install-macos.sh first" in result.stderr


@pytest.fixture
def interpreters(checkout):
    root, env = checkout
    bindir = root / ".venv" / "bin"
    bindir.mkdir(parents=True)
    probe = (
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    print(os.environ['TEST_LIBDIR'])\n"
        "else:\n"
        "    print(json.dumps(dict(interpreter=os.path.basename(sys.argv[0]), args=sys.argv[1:],\n"
        "        cwd=os.getcwd(), gl=os.environ.get('MUJOCO_GL'),\n"
        "        library_path=os.environ.get('DYLD_FALLBACK_LIBRARY_PATH'))))\n"
    )
    for name in ("python", "mjpython"):
        interpreter = bindir / name
        interpreter.write_text(probe)
        interpreter.chmod(0o755)
    env["TEST_LIBDIR"] = str(root / "Python lib")
    env["MUJOCO_GL"] = "egl"
    return root, env


@pytest.mark.parametrize(
    "args,interpreter",
    [
        (["sim", "--actuator", "Left Arm 1=-0.1"], "mjpython"),
        (["sim", "--headless", "--steps", "1"], "python"),
        (["sim", "--help"], "python"),
        (["serve", "--headless"], "python"),
        (["build-scene", "--scene", "house", "--render"], "python"),
    ],
)
def test_interpreter_arguments_and_renderer(interpreters, args, interpreter):
    root, env = interpreters
    result = run(interpreters, "run-macos.sh", *args)
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state["interpreter"] == interpreter
    assert state["args"] == ["-m", "abstract_stack.cli", *args]
    assert state["cwd"] == str(root)
    assert state["gl"] == "cgl"
    if interpreter == "mjpython":
        assert state["library_path"].split(":") == [
            env["TEST_LIBDIR"],
            "/usr/local/lib",
            "/usr/lib",
        ]
    else:
        assert state["library_path"] is None


def test_viewer_preserves_existing_library_path(interpreters):
    _, env = interpreters
    env["TEST_EXISTING_LIBRARY_PATH"] = "/custom/lib:/another/lib"
    result = run(interpreters, "run-macos.sh", "sim")
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state["library_path"].split(":") == [
        env["TEST_LIBDIR"],
        "/custom/lib",
        "/another/lib",
        "/usr/local/lib",
        "/usr/lib",
    ]
