"""Export the compiled MuJoCo scene's visual geometry for the browser robot twin.

The blob is derived from the exact ``MjModel`` being simulated: geom placement,
primitive sizes, colours and mesh buffers. Large visual meshes are simplified to a
face budget so phones can render the twin at display rate; the simplification only
affects this preview and never the physics model. Results are cached by content hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)

FORMAT = "web_teleop_model_v1"
# Scene primitives/products (0), robot visual meshes (2), house cutaway walls (5).
# Robot collision meshes (1) and camera markers (3) are omitted.
VISUAL_GROUPS = (0, 2, 5)
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)
GEOM_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "abstract_stack" / "web_teleop"


def _visual_geoms(model) -> list[int]:
    return [
        g
        for g in range(model.ngeom)
        if int(model.geom_group[g]) in VISUAL_GROUPS
        and int(model.geom_type[g]) in GEOM_TYPES
        and model.geom_rgba[g, 3] > 0
    ]


def model_hash(model, face_budget: int) -> str:
    digest = hashlib.sha256(f"{FORMAT}:{face_budget}".encode())
    for array in (
        model.geom_type,
        model.geom_group,
        model.geom_bodyid,
        model.geom_dataid,
        model.geom_size,
        model.geom_pos,
        model.geom_quat,
        model.geom_rgba,
        model.geom_matid,
        model.mat_rgba,
        model.body_parentid,
        model.mesh_vertnum,
        model.mesh_facenum,
        model.mesh_vert,
        model.mesh_face,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    for b in range(model.nbody):
        digest.update(model.body(b).name.encode() + b"\0")
    return digest.hexdigest()[:16]


def _simplify(vertices: np.ndarray, faces: np.ndarray, budget: int):
    if len(faces) <= budget:
        return vertices, faces
    import fast_simplification

    v, f = fast_simplification.simplify(
        vertices.astype(np.float32),
        faces.astype(np.int32),
        target_reduction=1 - budget / len(faces),
    )
    return v, f


def _colour(model, g: int) -> list[float]:
    rgba = model.geom_rgba[g]
    matid = int(model.geom_matid[g])
    # MuJoCo uses the material colour when the geom keeps the default rgba.
    if matid >= 0 and np.allclose(rgba, DEFAULT_RGBA):
        rgba = model.mat_rgba[matid]
    return [round(float(c), 4) for c in rgba]


def build_model_export(model, face_budget: int = 1500) -> tuple[dict, bytes]:
    geoms = _visual_geoms(model)
    mesh_ids = sorted({int(model.geom_dataid[g]) for g in geoms if model.geom_type[g] == 7})
    vertex_chunks, index_chunks, meshes, mesh_index = [], [], [], {}
    vertex_bytes = index_count = 0
    for mid in mesh_ids:
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        v, f = _simplify(model.mesh_vert[va : va + vn], model.mesh_face[fa : fa + fn], face_budget)
        v = np.ascontiguousarray(v, dtype="<f4")
        f = np.ascontiguousarray(f, dtype="<u4")
        mesh_index[mid] = len(meshes)
        meshes.append(
            {
                "name": model.mesh(mid).name,
                "vertex_offset": vertex_bytes,
                "vertex_count": int(len(v)),
                "index_offset": index_count * 4,
                "index_count": int(f.size),
            }
        )
        vertex_chunks.append(v.tobytes())
        index_chunks.append(f.tobytes())
        vertex_bytes += v.nbytes
        index_count += f.size
    for mesh in meshes:
        mesh["index_offset"] += vertex_bytes  # indices follow all vertex data
    blob = b"".join(vertex_chunks + index_chunks)
    manifest = {
        "format": FORMAT,
        "model_hash": model_hash(model, face_budget),
        "bodies": [
            {"name": model.body(b).name, "parent": int(model.body_parentid[b])}
            for b in range(model.nbody)
        ],
        "geoms": [
            {
                "name": model.geom(g).name,
                "body": int(model.geom_bodyid[g]),
                "group": int(model.geom_group[g]),
                "type": GEOM_TYPES[int(model.geom_type[g])],
                "size": [round(float(x), 6) for x in model.geom_size[g]],
                "pos": [round(float(x), 6) for x in model.geom_pos[g]],
                "quat": [round(float(x), 6) for x in model.geom_quat[g]],
                "rgba": _colour(model, g),
                "mesh": mesh_index.get(int(model.geom_dataid[g]), -1)
                if model.geom_type[g] == 7
                else -1,
            }
            for g in geoms
        ],
        "meshes": meshes,
        "cameras": [
            {
                "name": model.camera(c).name,
                "body": int(model.cam_bodyid[c]),
                "fovy_deg": float(model.cam_fovy[c]),
            }
            for c in range(model.ncam)
        ],
        "blob_bytes": len(blob),
        "face_count": index_count // 3,
    }
    return manifest, blob


def load_model_export(model, face_budget: int = 1500) -> tuple[dict, bytes]:
    """Return ``(manifest, blob)``, reusing a cached export with the same content hash."""
    key = model_hash(model, face_budget)
    directory = _cache_dir()
    manifest_path, blob_path = directory / f"{key}.json", directory / f"{key}.bin"
    try:
        manifest = json.loads(manifest_path.read_text())
        blob = blob_path.read_bytes()
        if manifest.get("model_hash") == key and manifest.get("blob_bytes") == len(blob):
            return manifest, blob
    except (OSError, ValueError):
        pass
    manifest, blob = build_model_export(model, face_budget)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(blob)
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    except OSError as error:  # caching is an optimisation only
        log.warning("could not cache web model export: %s", error)
    log.info(
        "web model export %s: %d geoms, %d faces, %.1f MB",
        key,
        len(manifest["geoms"]),
        manifest["face_count"],
        len(blob) / 1e6,
    )
    return manifest, blob
