# Git metadata and shared experiment sources

New experiments should record a Git revision and scoped dirty-source metadata,
then reference one shared content-addressed source bundle. They should not copy
the package into every run directory. Git identifies the committed baseline;
the bundle preserves the exact selected bytes, including uncommitted/untracked
policy code, and remains usable offline if the checkout or Git objects disappear.
Source hashes alone cannot reconstruct an uncommitted policy.

## Layout

```text
outputs/source_bundles/<content-id>.zip       # one immutable compressed source tree
outputs/cache/source_runtime/<content-id>/  # one verified runnable extraction
outputs/<experiment>/source_ref.json         # Git revision/status, ID, hashes, archive pointer
```

The ID hashes relative paths, exact file bytes and executable flags. Identical
source trees share the same ZIP and runtime even when invoked repeatedly.
Per-run metadata records that invocation's Git HEAD/tree and scoped dirty status;
Git identity is not guessed from a checkout containing unrelated snapshots.
Archive integrity and cached source bytes are checked before reuse. Modified or
missing archives/runtime trees fail rather than silently using current sources.
Runtime trees are read-only; use `PYTHONDONTWRITEBYTECODE=1` in frozen workers.
Unexpected cached bytecode, writable runtime sources and modified runtime
manifests are rejected before another worker starts.

Nested launchers may capture a smaller source scope from an already frozen
tree. They inherit the parent Git metadata only after their selected hashes
match that parent's bundle; `inherited_source_bundle_id` records that broader
original Git scope. Changed or unrelated trees cannot inherit its revision.

Select only the package and actual entrypoints/configuration needed by the
worker. No broad workspace diff is collected. Hidden files, environment files,
caches and unrelated output trees are excluded; symlinked sources are rejected.
Directory collection admits common source/config extensions, not model assets
or recordings. Capture is bounded to 10,000 files / 64 MiB and checked twice so
edits during freezing cause an explicit failure.

## Runner API

```python
from abstract_stack.source_store import freeze_source

frozen = freeze_source(
    root,
    paths=[
        "src/abstract_stack",
        "scripts/continue_stocking.py",
        "scripts/run_frozen_replanning.py",
        "pyproject.toml",
        "uv.lock",
    ],
    store=root / "outputs/source_bundles",
)
frozen.write_reference(output / "source_ref.json")
# Execute frozen.runtime_root / "scripts/continue_stocking.py" with:
# PYTHONPATH = frozen.runtime_root / "src"
# PYTHONDONTWRITEBYTECODE = "1"
# ABSTRACT_STACK_SOURCE_REFERENCE = output / "source_ref.json"
```

Run inputs, model/scene identity, controller mode/settings, dependency versions,
simulation timing and process outcomes still belong in each run's manifest.
A shared source bundle is not an episode, model checkpoint, action certificate,
or success result. Keep failed/unknown runs and their metadata.

`policy_provenance()` uses `ABSTRACT_STACK_SOURCE_REFERENCE` to retain the
original Git revision/dirty flags inside a frozen worker. It first checks that
the actual imported package's Python hashes match the bundle. A mismatched or
missing reference fails; it cannot label different live code with an old commit.

The default store follows the owning repository, including children running
inside cached snapshots. `ABSTRACT_STACK_SOURCE_STORE` can explicitly pin the
store. Captures outside any checkout use a per-user temporary store. Set a
durable store explicitly for long-running campaigns.

## Portable export and checkpoints

```python
from abstract_stack.source_store import resolve_source_reference

frozen.write_reference(export_directory / "source_ref.json", portable=True)
# Exports source_bundle.zip beside its reference, without modifying the run.
restored = resolve_source_reference(export_directory / "source_ref.json", cache=shared_cache)
# restored.runtime_root contains the exact selected files; no Git/network required.
```

Checkpoints remain portable by default. `save_checkpoint()` now embeds one
compressed `source_bundle.zip` and hashed `source_ref.json` instead of a copied
Python tree. MJB, complete integration state, controller/observer memory and
strict replay witnesses are unchanged. Inspection verifies the source archive
using the standard library, without importing MuJoCo or extracting Python files.

`portable_source=False` is an explicit internal optimization: only the source
reference is saved and loading requires its shared archive. Use the default
portable mode when moving a checkpoint outside the local store. A missing shared
archive is an error, never a fallback to the current checkout. The reference's
portability mode and source identity are hashed in the checkpoint manifest.

Historical checkpoints with `source_snapshot/` trees remain readable. Existing
outputs and their manifests are left untouched; this is a forward migration,
not cleanup that rewrites measured evidence. Some older launchers still copy
trees until migrated. New runners should use the shared API above.
