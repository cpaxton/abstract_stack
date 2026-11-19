"""Export the compiled MuJoCo scene's visual geometry for the browser robot twin.

The blob is derived from the exact ``MjModel`` being simulated: geom placement,
primitive sizes, colours, and mesh vertices/faces with MuJoCo's own compiled vertex
normals, so shading matches the native viewer. Meshes are exported at full resolution
by default; an optional per-mesh face budget simplifies them for constrained clients
(preview only, never the physics model). Results are cached by content hash.

Blob layout per mesh (4-byte aligned): float32 positions[n*3], int8 normals[n*3]
(normalised, padded to 4 bytes), then uint16 or uint32 triangle indices.
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

FORMAT = "web_teleop_model_v2"
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


def _simplified(vertices: np.ndarray, faces: np.ndarray, budget: int):
    import fast_simplification

    v, f = fast_simplification.simplify(
        vertices.astype(np.float32),
        faces.astype(np.int32),
        target_reduction=1 - budget / len(faces),
    )
    # Area-weighted smooth vertex normals for the simplified surface.
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals = np.zeros_like(v, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, f[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return v, normals, f


def _mesh_buffers(model, mid: int, face_budget: int):
    """Positions, unit normals and triangle indices for one compiled mesh."""
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    faces = model.mesh_face[fa : fa + fn]
    if face_budget and fn > face_budget:
        return _simplified(model.mesh_vert[va : va + vn], faces, face_budget)
    # MuJoCo indexes normals separately from positions; one vertex per unique
    # (position, normal) corner pair keeps the mesh indexed.
    normal_ids = model.mesh_facenormal[fa : fa + fn].astype(np.int64)
    pairs = faces.astype(np.int64) * (1 << 32) + normal_ids
    unique, inverse = np.unique(pairs.ravel(), return_inverse=True)
    positions = model.mesh_vert[va + (unique >> 32)]
    normals = model.mesh_normal[model.mesh_normaladr[mid] + (unique & 0xFFFFFFFF)]
    return positions, normals, inverse.reshape(-1, 3)


def _pad4(data: bytes) -> bytes:
    return data + b"\0" * (-len(data) % 4)


def _colour(model, g: int) -> list[float]:
    rgba = model.geom_rgba[g]
    matid = int(model.geom_matid[g])
    # MuJoCo uses the material colour when the geom keeps the default rgba.
    if matid >= 0 and np.allclose(rgba, DEFAULT_RGBA):
        rgba = model.mat_rgba[matid]
    return [round(float(c), 4) for c in rgba]


def build_model_export(model, face_budget: int = 0) -> tuple[dict, bytes]:
    geoms = _visual_geoms(model)
    mesh_ids = sorted({int(model.geom_dataid[g]) for g in geoms if model.geom_type[g] == 7})
    chunks, meshes, mesh_index, offset, face_count = [], [], {}, 0, 0
    for mid in mesh_ids:
        positions, normals, faces = _mesh_buffers(model, mid, face_budget)
        count = len(positions)
        index_type = "uint16" if count <= 0xFFFF else "uint32"
        position_bytes = np.ascontiguousarray(positions, dtype="<f4").tobytes()
        normal_bytes = _pad4(np.round(np.clip(normals, -1, 1) * 127).astype(np.int8).tobytes())
        index_bytes = _pad4(
            np.ascontiguousarray(faces, dtype="<u2" if index_type == "uint16" else "<u4").tobytes()
        )
        mesh_index[mid] = len(meshes)
        meshes.append(
            {
                "name": model.mesh(mid).name,
                "vertex_count": int(count),
                "position_offset": offset,
                "normal_offset": offset + len(position_bytes),
                "index_offset": offset + len(position_bytes) + len(normal_bytes),
                "index_count": int(faces.size),
                "index_type": index_type,
            }
        )
        chunks += [position_bytes, normal_bytes, index_bytes]
        offset += len(position_bytes) + len(normal_bytes) + len(index_bytes)
        face_count += len(faces)
    blob = b"".join(chunks)
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
        "face_count": face_count,
        "face_budget": face_budget,
    }
    return manifest, blob


def load_model_export(model, face_budget: int = 0) -> tuple[dict, bytes]:
    """Return ``(manifest, blob)``, reusing a cached export with the same content hash.

    ``face_budget=0`` exports full-resolution meshes.
    """
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
