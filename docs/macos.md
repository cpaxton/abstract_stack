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

The installer selects Python 3.11 by default (`PYTHON_VERSION=3.12` also selects
a supported interpreter). It installs the locked `ci` dependency group into
`.venv`, installs this project editable without Emet, fetches the pinned MABEL
model and meshes, and installs development hooks. Existing MABEL checkouts are
validated rather than overwritten. Both scripts work from any working directory;
relative output paths are resolved from the checkout root.

The launcher uses CGL for native Mac rendering. Interactive `sim` commands run
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

Press R in the interactive viewer to reset the episode. Use `--help` or
`sim --help` with the launcher for other CLI options. The `serve` command
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
