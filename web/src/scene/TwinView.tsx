import { Grid, OrbitControls, PerformanceMonitor, TransformControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  Group,
  Line,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshStandardMaterial,
  Quaternion,
  SphereGeometry,
  Vector3,
  type Object3D,
} from "three";
import { applyJog, emptyJog, RATES } from "../input/jog";
import { keyboardTurn, readKeyboard } from "../input/keyboard";
import { readTouch } from "../input/touch";
import type { TeleopClient } from "../net/TeleopClient";
import { SIDES, type Side } from "../net/protocol";
import { intent, stopBase } from "../state/intent";
import { useUi, type CameraPreset } from "../state/store";
import { applyPoses, type RobotModel } from "./robotModel";

// MuJoCo (Z-up) ↔ three.js (Y-up) for vectors in world space.
const toThree = (x: number, y: number, z: number) => new Vector3(x, z, -y);
const fromThree = (v: Vector3, out: Vector3) => out.set(v.x, -v.z, v.y);

const PALM_OFFSET = new Vector3(0, 0.045, 0.04); // matches retarget.PALM_OFFSET
const SIDE_COLOR: Record<Side, string> = { left: "#59b8ff", right: "#ffad4d" };
const UNREACHABLE = "#ff5d5d";

// Camera presets (MuJoCo world coordinates; MABEL faces -X, shelf in front).
const PRESETS: Record<
  CameraPreset,
  { eye: [number, number, number]; at: [number, number, number] }
> = {
  operator: { eye: [1.05, 0.0, 1.75], at: [-0.45, 0, 0.75] },
  side: { eye: [-0.35, 1.9, 1.2], at: [-0.35, 0, 0.8] },
  top: { eye: [-0.3, 0.0, 3.1], at: [-0.3, 0, 0.6] },
  front: { eye: [-1.9, 0.4, 1.35], at: [-0.2, 0, 0.85] },
};

interface Frames {
  chassisPos: Vector3;
  chassisQuat: Quaternion;
  palm: Record<Side, Vector3>; // world-frame control points (MuJoCo coordinates)
}

function useFrames(): Frames {
  return useMemo(
    () => ({
      chassisPos: new Vector3(),
      chassisQuat: new Quaternion(),
      palm: { left: new Vector3(), right: new Vector3() },
    }),
    [],
  );
}

function RobotTwin({
  model,
  client,
  frames,
}: {
  model: RobotModel;
  client: TeleopClient;
  frames: Frames;
}) {
  const chassis = model.bodyIndex("mabel_chassis");
  const palms = { left: model.bodyIndex("left_palm"), right: model.bodyIndex("right_palm") };
  useFrame(() => {
    if (!client.poses.sample()) return;
    applyPoses(model, client.poses.xpos, client.poses.xquat);
    const c = model.bodies[chassis];
    if (c) {
      frames.chassisPos.copy(c.position);
      frames.chassisQuat.copy(c.quaternion);
    }
    for (const side of SIDES) {
      const body = model.bodies[palms[side]];
      if (body)
        frames.palm[side].copy(PALM_OFFSET).applyQuaternion(body.quaternion).add(body.position);
    }
  }, -2);
  return <primitive object={model.root} />;
}

/** Translucent target marker with orientation triad; children of the MuJoCo world group. */
function makeGhost(side: Side) {
  const group = new Group();
  group.name = `ghost-${side}`;
  const material = new MeshStandardMaterial({
    color: SIDE_COLOR[side],
    emissive: SIDE_COLOR[side],
    emissiveIntensity: 0.35,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  group.add(new Mesh(new SphereGeometry(0.028, 24, 16), material));
  const axes = new BufferGeometry();
  // Palm-frame triad (x red, y green, z blue) to show target orientation.
  axes.setAttribute(
    "position",
    new BufferAttribute(
      new Float32Array([0, 0, 0, 0.09, 0, 0, 0, 0, 0, 0, 0.09, 0, 0, 0, 0, 0, 0, 0.09]),
      3,
    ),
  );
  axes.setAttribute(
    "color",
    new BufferAttribute(
      new Float32Array([
        1, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 1, 0.4, 0.3, 1, 0.4, 0.3, 0.6, 1, 0.3, 0.6, 1,
      ]),
      3,
    ),
  );
  const triad = new LineSegments(
    axes,
    new LineBasicMaterial({ vertexColors: true, depthTest: false }),
  );
  triad.renderOrder = 10;
  group.add(triad);
  const tether = new Line(
    new BufferGeometry().setAttribute("position", new BufferAttribute(new Float32Array(6), 3)),
    new LineBasicMaterial({ color: SIDE_COLOR[side], transparent: true, opacity: 0.7 }),
  );
  tether.frustumCulled = false;
  return { group, material, tether };
}

function Ghosts({
  model,
  frames,
  controlsEnabled,
}: {
  model: RobotModel;
  frames: Frames;
  controlsEnabled: boolean;
}) {
  const ghosts = useMemo(() => ({ left: makeGhost("left"), right: makeGhost("right") }), []);
  const activeArm = useUi((s) => s.activeArm);
  const gizmoMode = useUi((s) => s.gizmoMode);
  const status = useUi((s) => s.status);
  const dragging = useRef(false);

  useEffect(() => {
    for (const side of SIDES) {
      model.root.add(ghosts[side].group);
      model.root.add(ghosts[side].tether);
    }
    return () => {
      for (const side of SIDES) {
        model.root.remove(ghosts[side].group);
        model.root.remove(ghosts[side].tether);
      }
    };
  }, [model, ghosts]);
  const target: Object3D = ghosts[activeArm].group;

  const tmp = useMemo(() => ({ v: new Vector3(), q: new Quaternion(), inv: new Quaternion() }), []);
  useFrame(() => {
    const engaged = status?.state === "engaged";
    for (const side of SIDES) {
      const ghost = ghosts[side];
      const arm = intent.arms[side];
      const visible = intent.initialized && status !== null;
      ghost.group.visible = ghost.tether.visible = visible;
      if (!visible) continue;
      if (!(dragging.current && side === activeArm)) {
        // root → world: p_w = p_c + R_c p_r, q_w = q_c q_r
        ghost.group.position
          .copy(arm.position)
          .applyQuaternion(frames.chassisQuat)
          .add(frames.chassisPos);
        ghost.group.quaternion.copy(frames.chassisQuat).multiply(arm.quaternion);
      }
      const reachable = status?.arms[side]?.ik?.reachable ?? true;
      const color = reachable || !engaged ? SIDE_COLOR[side] : UNREACHABLE;
      ghost.material.color.set(color);
      ghost.material.emissive.set(color);
      ghost.material.opacity = engaged ? (side === activeArm ? 0.7 : 0.4) : 0.18;
      const positions = ghost.tether.geometry.getAttribute("position") as BufferAttribute;
      const p = frames.palm[side];
      const g = ghost.group.position;
      positions.setXYZ(0, p.x, p.y, p.z);
      positions.setXYZ(1, g.x, g.y, g.z);
      positions.needsUpdate = true;
    }
  }, -1);

  const onChange = () => {
    const object = ghosts[activeArm].group;
    const arm = intent.arms[activeArm];
    // world → root: p_r = R_c⁻¹ (p_w - p_c), q_r = q_c⁻¹ q_w
    tmp.inv.copy(frames.chassisQuat).invert();
    arm.position.copy(object.position).sub(frames.chassisPos).applyQuaternion(tmp.inv);
    arm.quaternion.copy(tmp.inv).multiply(object.quaternion).normalize();
  };

  if (!target || !controlsEnabled) return null;
  return (
    <TransformControls
      object={target}
      mode={gizmoMode}
      space="local"
      size={0.75}
      onMouseDown={() => (dragging.current = true)}
      onMouseUp={() => (dragging.current = false)}
      onObjectChange={onChange}
    />
  );
}

/** Keyboard/touch jog, integrated at display rate in the viewer's ground frame. */
function JogLoop({ frames, enabled }: { frames: Frames; enabled: boolean }) {
  const camera = useThree((s) => s.camera);
  const jog = useMemo(() => emptyJog(), []);
  const tmp = useMemo(
    () => ({
      dir: new Vector3(),
      mj: new Vector3(),
      fwd: new Vector3(),
      right: new Vector3(),
      inv: new Quaternion(),
    }),
    [],
  );
  useEffect(() => {
    if (!enabled) stopBase();
  }, [enabled]);
  useFrame((_, delta) => {
    if (!enabled || !intent.initialized) return;
    const ui = useUi.getState();
    jog.move.set(0, 0, 0);
    jog.rotate.set(0, 0, 0);
    jog.grip = jog.headYaw = jog.headPitch = jog.lift = 0;
    jog.fine = false;
    readKeyboard(jog);
    readTouch(jog);
    if (ui.driveMode) {
      // Robot-centric driving: stick/WASD up = robot forward, Q/E or stick-x = turn.
      const limits = ui.status?.base?.limits ?? { linear_mps: 0.45, angular_radps: 0.9 };
      const scale = jog.fine ? RATES.fineScale : 1;
      const clamp1 = (v: number) => Math.max(-1, Math.min(1, v));
      intent.base.forward = clamp1(jog.move.y) * limits.linear_mps * scale;
      intent.base.left = -clamp1(jog.move.x) * limits.linear_mps * scale;
      intent.base.turn = clamp1(jog.rotate.z + keyboardTurn()) * limits.angular_radps * scale;
      // Head, lift and grip keys keep working while driving.
      jog.move.set(0, 0, 0);
      jog.rotate.set(0, 0, 0);
    } else {
      stopBase();
    }
    if (
      jog.move.lengthSq() === 0 &&
      jog.rotate.lengthSq() === 0 &&
      !jog.grip &&
      !jog.headYaw &&
      !jog.headPitch &&
      !jog.lift
    ) {
      return;
    }
    jog.move.clampScalar(-1, 1);
    jog.rotate.clampScalar(-1, 1);
    if (useUi.getState().mainView === "camera") {
      // Looking through the robot's head: "forward" is where the head points
      // (root frame: forward = -X, neck yaw positive turns toward the robot's left = -Y).
      const yaw = intent.head.yaw;
      tmp.fwd.set(-Math.cos(yaw), -Math.sin(yaw), 0);
    } else {
      // Viewer forward projected on the ground, expressed in the robot root frame.
      camera.getWorldDirection(tmp.dir);
      fromThree(tmp.dir, tmp.mj);
      tmp.mj.z = 0;
      if (tmp.mj.lengthSq() < 1e-6) tmp.mj.set(-1, 0, 0);
      tmp.mj.normalize();
      tmp.inv.copy(frames.chassisQuat).invert();
      tmp.fwd.copy(tmp.mj).applyQuaternion(tmp.inv);
      tmp.fwd.z = 0;
      tmp.fwd.normalize();
    }
    tmp.right.set(tmp.fwd.y, -tmp.fwd.x, 0); // forward × up
    applyJog(Math.min(delta, 0.1), jog, useUi.getState().activeArm, tmp.fwd, tmp.right);
  });
  return null;
}

/**
 * Presets are placed relative to the robot (its current position and heading), and
 * the orbit camera then follows the chassis as it drives so the robot stays framed.
 */
function CameraRig({ frames }: { frames: Frames }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as unknown as {
    target: Vector3;
    update: () => void;
  } | null;
  const preset = useUi((s) => s.camera);
  const nonce = useUi((s) => s.cameraNonce);
  const pending = useRef(true);
  const last = useMemo(() => new Vector3(Number.NaN, 0, 0), []);
  const tmp = useMemo(() => ({ v: new Vector3(), d: new Vector3() }), []);
  useEffect(() => {
    pending.current = true;
  }, [preset, nonce]);
  useFrame(() => {
    const chassisKnown = frames.chassisPos.lengthSq() > 0 || frames.chassisPos.z !== 0;
    if (pending.current && chassisKnown) {
      pending.current = false;
      const { eye, at } = PRESETS[preset];
      const place = (p: [number, number, number]) =>
        toThree(
          ...(tmp.v
            .set(p[0], p[1], p[2])
            .applyQuaternion(frames.chassisQuat)
            .add(frames.chassisPos)
            .toArray() as [number, number, number]),
        );
      const target = place([at[0], at[1], at[2] - 0.078]);
      camera.position.copy(place([eye[0], eye[1], eye[2] - 0.078]));
      camera.lookAt(target);
      if (controls) {
        controls.target.copy(target);
        controls.update();
      }
      last.copy(frames.chassisPos);
      return;
    }
    if (!chassisKnown || Number.isNaN(last.x)) return;
    // Follow chassis translation (MuJoCo → three.js), keeping the user's orbit.
    const dx = frames.chassisPos.x - last.x;
    const dy = frames.chassisPos.y - last.y;
    if (dx * dx + dy * dy < 1e-10) return;
    const delta = toThree(dx, dy, 0);
    camera.position.add(delta);
    if (controls) {
      controls.target.add(delta);
      controls.update();
    }
    last.copy(frames.chassisPos);
  }, -1);
  return null;
}

export function TwinView({ model, client }: { model: RobotModel | null; client: TeleopClient }) {
  const frames = useFrames();
  const status = useUi((s) => s.status);
  const role = useUi((s) => s.role);
  const quality = useUi((s) => s.quality);
  const set = useUi((s) => s.set);
  const operating = role === "operator" && status?.state === "engaged";
  const driveMode = useUi((s) => s.driveMode);
  const high = quality === "high";
  const dpr = high ? Math.min(2, window.devicePixelRatio || 1) : 1;
  const initial = PRESETS.operator;
  return (
    <Canvas
      className="twin-canvas"
      shadows={high ? "percentage" : false}
      dpr={dpr}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      camera={{ fov: 50, near: 0.03, far: 60, position: toThree(...initial.eye) }}
    >
      {/* Keep input/command timing healthy on slow GPUs: shed shadows and resolution. */}
      <PerformanceMonitor onDecline={() => set({ quality: "low" })} flipflops={2} />
      <color attach="background" args={["#0d1117"]} />
      <fog attach="fog" args={["#0d1117", 6, 18]} />
      <hemisphereLight args={["#dfe9f5", "#1b2027", 0.9]} />
      <directionalLight
        position={[2.5, 5, 2]}
        intensity={2.1}
        castShadow={high}
        shadow-mapSize={[1024, 1024]}
        shadow-camera-left={-3}
        shadow-camera-right={3}
        shadow-camera-top={3}
        shadow-camera-bottom={-3}
        shadow-bias={-0.0004}
      />
      <directionalLight position={[-3, 2.5, -2]} intensity={0.6} />
      <Grid
        position={[0, 0.002, 0]}
        args={[20, 20]}
        cellSize={0.1}
        cellThickness={0.6}
        cellColor="#2a323d"
        sectionSize={1}
        sectionThickness={1}
        sectionColor="#3a4654"
        fadeDistance={14}
        infiniteGrid
      />
      {model && <RobotTwin model={model} client={client} frames={frames} />}
      {model && <Ghosts model={model} frames={frames} controlsEnabled={operating && !driveMode} />}
      <JogLoop frames={frames} enabled={operating} />
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.12}
        maxDistance={12}
        minDistance={0.3}
      />
      <CameraRig frames={frames} />
    </Canvas>
  );
}
