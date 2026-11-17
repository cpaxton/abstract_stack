# Native macOS simulator

The full Emet dependency stack includes Linux GPU and camera packages that cannot
be installed unchanged on macOS. Use the lightweight simulator environment for
local MuJoCo development, MABEL inspection, physics tests, and scene rendering.
Emet/ZMQ integration, its learning stack, and the upstream VR/controller tools
are not included.

## Install and run

Install Git and [uv](https://docs.astral.sh/uv/), then run from this checkout:

```bash
./scripts/install-macos.sh
./scripts/run-macos.sh sim
./scripts/run-macos.sh sim --scene house
```

Supported Python versions are 3.10–3.12 (`>=3.10,<3.13`); 3.13 and newer are
excluded. The installer selects 3.11 by default; set `PYTHON_VERSION=3.10` or
`PYTHON_VERSION=3.12` to choose another supported interpreter.
It installs the locked `ci` dependency group into
`.venv`, installs this project editable without Emet, fetches the pinned MABEL
model and meshes, and installs development hooks. Existing MABEL checkouts are
validated rather than overwritten. Both scripts work from any working directory;
relative output paths are resolved from the checkout root.

The launcher defaults `MUJOCO_GL` to `cgl` for every subcommand, including headless
and offscreen runs. An explicit value is preserved; for example,
`MUJOCO_GL=glfw ./scripts/run-macos.sh sim` selects GLFW. CGL rendering requires
access to macOS graphics services even when no viewer window is opened.
Interactive `sim` commands run
through MuJoCo's `mjpython` launcher so Cocoa rendering can use the main thread.
It also adds the interpreter's library directory to the process-local
`DYLD_FALLBACK_LIBRARY_PATH`, allowing uv-managed Python's `libpython` to load.
Headless commands use ordinary Python. Shell startup files are not changed.

```bash
# Physics-only run with the complete MABEL model.
./scripts/run-macos.sh sim --headless --steps 500 --json outputs/smoke.json

# Offscreen house rendering and scene export using CGL rather than EGL.
./scripts/run-macos.sh build-scene --scene house --render --output-dir outputs/house

# Physics tests; the Emet adapter test skips in this environment.
MUJOCO_GL=cgl ./scripts/test.sh

# Development checks without re-synchronizing the full Emet dependency stack.
uv run --no-sync pre-commit run --all-files
uv build
```

Press R in the interactive viewer to reset the episode. Use `--help` with the
launcher for launcher usage, or `sim --help` for simulation CLI options. Launcher
help is available before installation and on other operating systems.
The `serve` command
requires Emet and is unavailable in this lightweight environment.

## Keeping the environment lightweight

Use `scripts/install-macos.sh` again to synchronize this environment. The root
`install.sh`, plain `uv sync`, and an ordinary `uv run` request the full project
dependencies, including Emet. Use `uv run --no-sync` when running tools already
installed in `.venv`. Direct `.venv/bin/python` commands can use
`MUJOCO_GL=cgl` for offscreen rendering; use the launcher for the native viewer.

The project still declares Emet as a dependency, so `uv pip check` reports it
missing in this installation. This follows the existing lightweight CI pattern;
it does not validate Emet integration. No dependency declarations or lock entries
are changed by these scripts.

## Validated configuration

Validated on an Apple Silicon Mac with macOS 27.0.1, Python 3.11.15,
MuJoCo 3.17.0, and NumPy 1.26.4. Validation covers real-MABEL physics tests,
a bounded native viewer session, and the eight house render cameras. It does not
establish Intel Mac compatibility or autonomous manipulation success. Some floor
speckling was visible in house previews; its cause has not been established.

Linux CI runs ShellCheck on both scripts and portable subprocess tests for
launcher help, platform/setup errors, interpreter selection, argument forwarding,
CGL defaults, explicit backend overrides, and preservation of existing library
search paths. A separate `macos-15` CI job invokes the actual installer, runs
native MABEL physics tests and headless launcher smoke checks, and exports the
house scene. Linux CI requires the rendered house previews. CGL rendering and
the interactive viewer remain manual checks on a Mac with graphics access.

The first native CI run passed installation, the headless launcher and 62 tests
(one expected Emet adapter skip), then failed CGL rendering with
`invalid pixel format` on the hosted macOS 15.8.2 VM. That
[failed attempt](https://github.com/cpaxton/abstract_stack/actions/runs/44004661921)
is retained; it does not invalidate the earlier hardware rendering evidence or
establish CGL support on hosted runners. The required Mac job checks physics and
scene export without claiming a rendering pass.

The native CI job excludes tests marked `rendering`, including the dual-camera
MP4 integration test added with the controller stack. Those tests remain required
in Linux CI and run by default on a real Mac when its graphics services and
encoding tools are available. Physics and non-rendering robot tests still run on
the hosted Mac.

Runner details follow [GitHub's hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The `mjpython` viewer requirement is documented by
[MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).
