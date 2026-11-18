import {
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CapsuleGeometry,
  CylinderGeometry,
  Group,
  Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
  SphereGeometry,
  type Material,
} from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";

export interface ModelManifest {
  format: "web_teleop_model_v1";
  model_hash: string;
  bodies: { name: string; parent: number }[];
  geoms: {
    name: string;
    body: number;
    group: number;
    type: "plane" | "sphere" | "capsule" | "ellipsoid" | "cylinder" | "box" | "mesh";
    size: [number, number, number];
    pos: [number, number, number];
    quat: [number, number, number, number];
    rgba: [number, number, number, number];
    mesh: number;
  }[];
  meshes: {
    name: string;
    vertex_offset: number;
    vertex_count: number;
    index_offset: number;
    index_count: number;
  }[];
  cameras: { name: string; body: number; fovy_deg: number }[];
  blob_bytes: number;
  face_count: number;
}

export interface RobotModel {
  manifest: ModelManifest;
  /** Z-up MuJoCo world, rotated into three.js' Y-up world. Add MuJoCo-frame children here. */
  root: Group;
  /** One group per MuJoCo body; poses are world-frame (flat, not a hierarchy). */
  bodies: Group[];
  bodyIndex: (name: string) => number;
  /** Geom groups (e.g. house cutaway walls = 5) for visibility toggles. */
  groups: Map<number, Mesh[]>;
  dispose: () => void;
}

const CREASE_ANGLE = (35 * Math.PI) / 180;

function primitive(geom: ModelManifest["geoms"][number]): BufferGeometry {
  const [a, b, c] = geom.size;
  switch (geom.type) {
    case "box":
      return new BoxGeometry(2 * a, 2 * b, 2 * c);
    case "sphere":
      return new SphereGeometry(a, 24, 16);
    case "ellipsoid":
      return new SphereGeometry(1, 24, 16).scale(a, b, c);
    case "capsule":
      // MuJoCo capsules/cylinders run along local Z; three.js builds them along Y.
      return new CapsuleGeometry(a, 2 * b, 6, 16).rotateX(Math.PI / 2);
    case "cylinder":
      return new CylinderGeometry(a, a, 2 * b, 24).rotateX(Math.PI / 2);
    case "plane":
      return new PlaneGeometry(a > 0 ? 2 * a : 40, b > 0 ? 2 * b : 40);
    default:
      throw new Error(`unsupported geom type ${geom.type}`);
  }
}

export async function loadRobotModel(
  origin: string,
  token: string | null,
  signal?: AbortSignal,
): Promise<RobotModel> {
  const query = token ? `token=${encodeURIComponent(token)}` : "";
  const manifestResponse = await fetch(new URL(`/api/model?${query}`, origin), { signal });
  if (!manifestResponse.ok) throw new Error(`model manifest: HTTP ${manifestResponse.status}`);
  const manifest = (await manifestResponse.json()) as ModelManifest;
  const blobResponse = await fetch(
    new URL(`/api/model.bin?v=${manifest.model_hash}${query ? `&${query}` : ""}`, origin),
    { signal },
  );
  if (!blobResponse.ok) throw new Error(`model data: HTTP ${blobResponse.status}`);
  const blob = await blobResponse.arrayBuffer();
  if (blob.byteLength !== manifest.blob_bytes) throw new Error("model data size mismatch");

  const meshGeometries = manifest.meshes.map((m) => {
    const geometry = new BufferGeometry();
    geometry.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(blob, m.vertex_offset, m.vertex_count * 3), 3),
    );
    geometry.setIndex(new BufferAttribute(new Uint32Array(blob, m.index_offset, m.index_count), 1));
    // Simplified CAD meshes: crease hard edges, smooth curved surfaces.
    const creased = toCreasedNormals(geometry, CREASE_ANGLE);
    geometry.dispose();
    creased.computeBoundingSphere();
    return creased;
  });

  const materials = new Map<string, MeshStandardMaterial>();
  const material = (rgba: number[]): Material => {
    const key = rgba.join(",");
    let m = materials.get(key);
    if (!m) {
      const [r = 0.5, g = 0.5, b = 0.5, alpha = 1] = rgba;
      m = new MeshStandardMaterial({
        roughness: 0.62,
        metalness: 0.08,
        transparent: alpha < 1,
        opacity: alpha,
      });
      m.color.setRGB(r, g, b, "srgb");
      materials.set(key, m);
    }
    return m;
  };

  const root = new Group();
  root.name = "mujoco-world";
  root.rotation.x = -Math.PI / 2; // MuJoCo Z-up → three.js Y-up
  const bodies = manifest.bodies.map((body) => {
    const group = new Group();
    group.name = body.name;
    root.add(group);
    return group;
  });
  const groups = new Map<number, Mesh[]>();
  const owned: BufferGeometry[] = [...meshGeometries];
  for (const geom of manifest.geoms) {
    let geometry: BufferGeometry;
    if (geom.type === "mesh") {
      const shared = meshGeometries[geom.mesh];
      if (!shared) continue;
      geometry = shared;
    } else {
      geometry = primitive(geom);
      owned.push(geometry);
    }
    const mesh = new Mesh(geometry, material(geom.rgba));
    mesh.name = geom.name;
    mesh.position.set(...geom.pos);
    const [w, x, y, z] = geom.quat;
    mesh.quaternion.set(x, y, z, w);
    mesh.castShadow = geom.type !== "plane";
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    bodies[geom.body]?.add(mesh);
    const list = groups.get(geom.group) ?? [];
    list.push(mesh);
    groups.set(geom.group, list);
  }

  const index = new Map(manifest.bodies.map((b, i) => [b.name, i]));
  return {
    manifest,
    root,
    bodies,
    groups,
    bodyIndex: (name) => index.get(name) ?? -1,
    dispose: () => {
      owned.forEach((g) => g.dispose());
      materials.forEach((m) => m.dispose());
    },
  };
}

/** Apply world-frame body poses (MuJoCo xpos / xquat wxyz) to the body groups. */
export function applyPoses(model: RobotModel, xpos: Float32Array, xquat: Float32Array): void {
  const n = Math.min(model.bodies.length, xpos.length / 3);
  for (let i = 0; i < n; i++) {
    const body = model.bodies[i]!;
    body.position.set(xpos[3 * i]!, xpos[3 * i + 1]!, xpos[3 * i + 2]!);
    body.quaternion.set(xquat[4 * i + 1]!, xquat[4 * i + 2]!, xquat[4 * i + 3]!, xquat[4 * i]!);
  }
}
