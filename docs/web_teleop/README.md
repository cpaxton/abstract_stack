# Web teleoperation (browser, phone, tablet)

Teleoperate MABEL in `ShelfStockingEnv` from a browser. The operator sees the
robot's stereo head cameras (`zed_left`/`zed_right`) and an interpolated 3D twin
of the robot and scene. They drag or jog a target for each palm, and the host
turns those targets into bounded actuator commands.
Desktop uses mouse and keyboard; phones and tablets get on-screen joysticks.
The same client is meant to be wrapped later as an iOS/Android app (Capacitor)
and extended with WebXR.

Status: simulation only. Session recording, wrist-camera video and WebXR are not
implemented yet (see [Limitations](#limitations)).

## Run

```bash
# Python host (needs the `teleop` extra; the Mac installer's `ci` group includes it)
uv sync --extra teleop          # or: ./scripts/install-macos.sh
cd web && npm ci && npm run build && cd ..

# Local machine
./scripts/run-macos.sh teleop                     # or: abstract-stack teleop
# → prints  Web teleop: http://localhost:8080/?token=<generated>

# Phone/tablet on the same trusted LAN
abstract-stack teleop --host 0.0.0.0 --port 8080
# → also prints  Web teleop (LAN, phone/tablet): http://192.168.x.y:8080/?token=<token>
```

Scan the QR code the host prints with the phone camera, or open the printed
LAN URL. On macOS, allow incoming connections for
Python if the firewall asks; otherwise the phone cannot connect. On headless
Linux, set `MUJOCO_GL=egl` so the video thread can render offscreen.

The access token is created once and stored in
`~/.config/abstract_stack/web_teleop_token`, so saved links and home-screen
shortcuts keep working across restarts. Tokens look like `k7mz-q2xp-9vda`:
they avoid look-alike characters and match case-insensitively, so they are easy
to type on a phone. Use `--new-token` to rotate it, or
`--token` to set one. The client remembers the token and strips it from the
address bar. A wrong token shows "Access needed" instead of reconnecting. `--insecure-no-token` disables
it on a trusted development machine. A second browser joins as a view-only
spectator while an operator holds control; `?role=spectator` requests that
explicitly. `?quality=low` forces the light render path, which the client also
switches to automatically when the frame rate drops. Low quality also loads a
simplified twin mesh (800 faces per mesh); the default is full resolution.

Useful options: `--scene`, `--seed`, `--spawn-backoff` (default 0.15 m; see
below), `--time-limit` (default 3600 simulated
seconds), `--input-timeout` (default 0.15 s), `--face-budget` (0 = full-resolution
twin), `--no-video`, `--video-fps` (default 20), `--video-size` (default
640x480 per eye) and `--video-quality` (JPEG quality, default 80).

Front-end development with hot reload proxies `/api` and `/teleop` to a
running host:

```bash
abstract-stack teleop --port 8080 --insecure-no-token   # terminal 1
cd web && npm run dev                                    # terminal 2, open the printed URL
```

## Operating

| Action | Desktop | Touch |
| --- | --- | --- |
| Engage | <kbd>Enter</kbd> or **Engage** | **Engage** |
| Pause (always available) | <kbd>Esc</kbd> / <kbd>Space</kbd> | **Pause** |
| Move hand (relative to the view) | <kbd>W A S D</kbd>, <kbd>Q</kbd>/<kbd>E</kbd> down/up, or drag the gizmo | left stick; right stick up/down |
| Rotate hand | <kbd>I K</kbd> pitch, <kbd>J L</kbd> yaw, <kbd>U O</kbd> roll, or gizmo in Rotate mode (<kbd>Y</kbd>) | right stick left/right (yaw) |
| Grip | <kbd>[</kbd>/<kbd>]</kbd>, <kbd>G</kbd> toggle, slider | **Open** / **Close** (hold) |
| Switch hand / drive | <kbd>Tab</kbd> cycles left hand → right hand → drive | control chip |
| Drive the base (drive mode) | <kbd>W A S D</kbd> relative to the robot, <kbd>Q</kbd>/<kbd>E</kbd> turn | left stick drives, right stick turns |
| Head | follows the active hand automatically; arrows or drag/click the camera view for free look; click the 3D view to look at a spot; <kbd>F</kbd> or the Look chip follows again; <kbd>Home</kbd> re-centres | same gestures by touch; tap the Look chip to follow again |
| Height (lift), Lean (torso) | Height/Lean rails on the right edge, <kbd>PgUp</kbd>/<kbd>PgDn</kbd> | rails on the left edge |
| Fine motion | hold <kbd>Shift</kbd> | **Fine** toggle |
| Ready pose | <kbd>R</kbd> | top-bar button |
| Main view 3D ⇄ head camera | <kbd>V</kbd>, **3D / Camera** switch, click the PiP | switch, tap the PiP |
| Head-camera eye: left / both / right | <kbd>B</kbd> | switch |
| Camera presets / help | <kbd>1</kbd>–<kbd>4</kbd> / <kbd>?</kbd> | panel |

Start with **Ready pose**. The reset posture has both arms straight ahead, an
elbow singularity where Cartesian control responds poorly. Ready bends the
elbows and brings the palms about 0.4 m in front of the chassis. Each glowing
target shows where that palm is commanded to go. A tether connects it to the
measured palm, and the target turns red when IK cannot reach it. Leaving the
tab pauses immediately.

**First run.** A four-card guide (Hands, Look, Drive, Body) appears once for a
new operator, with text for the device in use. The "?" button in the top bar
reopens it.

**Looking.** By default the head follows the hand you control, and looks ahead
while driving. The host aims the head camera at the gaze point with the existing
`HeadGaze` solver. Any manual input (arrows, dragging or tapping the camera view,
the head sliders) switches to Free look, starting from the current angles.
Clicking the 3D view looks at that spot, shown by a marker. The Look chip shows
the mode, and tapping it (or pressing <kbd>F</kbd>) follows the hand again.

**Body.** The Height and Lean rails stay at the screen edge while engaged. MABEL's
torso only leans; the rails' "Turn → Drive" link switches to drive mode,
because the robot turns by driving.

**Driving.** In drive mode, movement input drives MABEL's swerve base relative to
the robot's own heading; the 3D view follows the robot. The base ramps up gently
(0.45 m/s and 0.9 rad/s maximum). It brakes as soon as input stops, control
pauses, the connection drops, or the chassis tilts past 8°.

**Spawn backoff.** Every bundled scene spawns MABEL with its fingers inside the
shelf front. The hooked hands then pull the robot over as soon as the arms or the
base move (measured: 11–55° tilt). Teleop therefore starts MABEL 0.15 m further
back (`--spawn-backoff 0` keeps the scene placement). The scenes themselves are
unchanged, and the operator can drive up to the shelf.

The head camera and the 3D twin swap between the main view and a
picture-in-picture. `zed_left` and `zed_right` are a 63 mm stereo pair, so the
eyes differ by a small horizontal parallax; **Both** shows them side by side. In
the camera view, <kbd>W</kbd> moves the hand in the direction the head is facing.

## Architecture

```
Browser (web/)                                  Host (src/abstract_stack/web_teleop/)
 TeleopClient ── WS /teleop/control (JSON) ──►  server.py  Starlette/uvicorn, auth, fan-out
   commands 30 Hz, control, ping   ◄── status 10 Hz, acks, pong
 PoseBuffer  ◄── WS /teleop/state (binary WTS1, 50 Hz body poses)
 robotModel  ◄── GET /api/model (+ model.bin)   geometry.py  visual geoms from the live MjModel
 CameraView  ◄── WS /teleop/video (binary QTP1 stereo JPEG pairs)   video.py  render thread
 TwinView (three.js / react-three-fiber)        core.py      sim-owner thread + session state machine
 input/ (keyboard, touch, gizmo → intent)       retarget.py  intents → PoseIK → MabelController
```

- **Single command writer.** One thread in `core.TeleopCore` owns the
  environment. It is the only code that writes `data.ctrl` or calls
  `env.step`/`env.reset`. Network handlers only validate and enqueue; reset
  and ready run on the owner thread and are acknowledged after they apply.
- **Retargeting** (`retarget.WebRetargeter`). Each tick, for every arm:
  root-frame palm target → per-side `PoseIK` on scratch data (`ik.py`, now with
  a `palm` argument) → arm/wrist references → `MabelController` (rate limits,
  robot-gravity feedforward) → `validate_targets`. IK warm-starts from the
  previous solution and keeps refining until it converges. Lift is the
  operator's own channel and is pinned during both arms' IK. Grip maps one
  fraction to the native flexion ranges and keeps the scripted controller's
  thumb preshape. Head and torso are clamped and rate-limited.
- **Base** (`base.SwerveBase`). The base receives a robot-frame twist and turns
  it into native steer angles and wheel speeds. Module positions, zero-steer
  rolling directions and wheel radius come from the compiled model; the rolling
  directions were verified against MuJoCo Jacobians. Modules aim before the base
  accelerates, reverse instead of turning past 90°, and drive only the component
  along their measured rolling direction. With clear surroundings, a 3 s,
  0.3 m/s command travels 0.79 m (ramps included) with no sideways drift and no
  tilt.
- **Nominal, applied and measured are kept separate.** Status reports IK
  position error (m) separately from orientation error (rad), the measured
  palm pose, and which joints were rate- or target-limited.
- **Delivery is latest-wins.** Each socket's sender transmits the newest
  frame/status when it is ready, so a slow peer skips frames instead of
  queueing them or slowing physics.
- **The twin is the real model.** `geometry.py` exports visual geom groups 0,
  2 and 5 (scene, robot visuals, house cutaway walls) from the compiled
  `MjModel`. Meshes are full resolution with MuJoCo's own vertex normals (1.3M
  faces, 12.8 MB gzipped for the stocking scene; 16-bit indices where
  possible). They are cached by content hash, compressed once, and served
  immutable, so browsers download them once. `detail=low` serves an 800
  faces-per-mesh preview for weak devices. Simplification affects the preview
  only, never physics.
- **Head-camera video** (`video.py`). A render thread owns its own GL context,
  `mujoco.Renderer` and scratch `MjData`. After each tick the sim owner only
  copies `qpos`/mocap into a snapshot. The render thread forwards kinematics
  and renders both eyes from that one state, with no physics step between
  eyes. Each pair goes out as one `QTP1` message, the Quest contract's stereo
  framing: `"QTP1"`, three big-endian uint32 lengths, JSON metadata, left JPEG,
  right JPEG. Rendering runs only while a client is subscribed, and the client
  unsubscribes when no camera view is visible. Measured on an Apple-silicon
  Mac: 20 pairs/s at 640×480 per eye, about 37 KB per pair (6 Mbit/s),
  physics real-time factor 1.0. The client decodes off the main thread
  (`createImageBitmap`), keeps only the newest pair, swaps both eyes together,
  and flags video older than 300 ms as stale.
- **Client performance.** 50 Hz poses go into a jitter buffer that plays back
  about 40 ms behind the fastest observed arrival and interpolates at display
  rate. High-rate data never touches React state; panels re-render at most
  10 Hz.

## Protocol (`web_teleop_v1`)

This profile reuses the Quest Teleop v1 envelope and state machine
([../quest_teleop/README.md](../quest_teleop/README.md)): `schema_version: 1`,
`type`, a UUID `session_id` per connection, strictly increasing `sequence` and
`event_id`, idempotent acknowledgements, and rejection of unknown fields,
non-finite numbers, non-unit quaternions and payloads over 16 KiB.
Implementations: `src/abstract_stack/web_teleop/protocol.py` and
`web/src/net/protocol.ts`.

1. Control socket: `hello {profile:"web_teleop_v1", role}` → `hello_ack {role,
   model_hash, calibration_generation, budgets, status}`.
2. State socket `/teleop/state`: `state_bind {session_id}`, then binary frames. Each frame is
   a 28-byte little-endian header (`"WTS1"`, tick u32, sim_time f64,
   host_time f64, nbody u16, flags u16), then float32 `xpos[nbody·3]` and
   `xquat[nbody·4]` (wxyz, world frame).
3. Video socket `/teleop/video`: `video_bind {session_id}`, then `QTP1` stereo
   pairs. `hello_ack.video` describes the cameras, size, rate and field of view,
   or is `null` with `--no-video`.
4. `command`: absolute intents with `calibration_generation`, `target_epoch`,
   `arms.{left,right}` (palm control-point `position_m` and `quaternion_wxyz`
   in the `mabel_chassis` frame, plus `grip` ∈ [0,1], or null to hold), and
   `head`, `lift_m`, `torso_rad` (null holds). `head` is either
   `{yaw_rad, pitch_rad}` or `{look_at_m: [x,y,z]}`, a chassis-frame gaze point
   the host tracks. An optional `base`
   `{forward_mps, left_mps, turn_radps}` is a velocity intent: null stops the
   base, and it only applies while engaged with fresh input. Absolute targets make every
   message self-contained, so a lost message never accumulates drift.
5. `control {event_id, calibration_generation, action}` with actions
   `engage | pause | reset | ready` → `control_ack`.
6. `ping {t0}` → `pong {t0, t1}` for round-trip display.

**Target epoch.** Whenever the host moves arm references on its own (engage,
ready, reset), it increments `target_epoch` and publishes `arm_targets`. The
client rebases its targets on those, and the host ignores arm intents from an
older epoch. That way a stale ghost can never yank the robot back.

**State machine.** Disconnected → Ready (operator connected) → Engaged
(deliberate engage) ⇄ Paused (operator pause, tab hidden, input older than
`--input-timeout`, episode time limit). Fault covers non-finite physics or an
invalid command and needs an explicit reset. There is no automatic resume.
Pause is honoured even with an obsolete generation. Reset increments the
generation and drops commands from older generations. While paused or
disconnected, the controller holds the current references, with gravity
feedforward still applied.

## Security and networks

Defaults are safe for a trusted LAN: a random access token is required for the
API and both sockets, WebSocket `Origin` must match the host (add others with
`--allow-origin`), payload sizes are bounded, and only one operator holds
control. For use beyond the LAN, put the host behind a TLS reverse proxy (for
example Caddy with `reverse_proxy 127.0.0.1:8080`; WebSockets pass through)
and keep the token. The client switches to `wss:` automatically on HTTPS.
Expect to raise `--input-timeout` on high-latency links. The 150 ms default
is the Quest contract's observation-age budget.

## Tests

```bash
MUJOCO_GL=cgl ./scripts/test.sh tests/test_web_teleop.py -q   # protocol, state machine, tracking, sockets
cd web && npm run lint && npm run typecheck && npm test && npm run build
cd web && npx playwright install chromium && npm run e2e   # launches a real host on :8799
```

The native tests check:

- lease and engage preconditions;
- that missing input pauses without auto-resume;
- that both arms track Cartesian targets from Ready to within 1 cm in native
  physics, with every applied control inside actuator limits;
- stale epochs and generations;
- model-export consistency;
- the WebSocket handshake, token check and state stream;
- `QTP1` framing, and a real stereo render streamed over the video socket
  (both eyes, with measurable parallax).

The browser e2e test exercises connect, keyboard jog, pause on input loss,
stereo video and the touch layout. Headless Chromium rasterises WebGL on the
CPU, so the tests use `?quality=low` and the e2e host runs with
`--input-timeout 0.8`.

## Limitations

- **Video is JPEG over WebSocket.** That is simple and low-latency on a LAN,
  but uses about 6 Mbit/s. For internet use, H.264 over WebRTC with
  congestion control is the planned upgrade. Wrist cameras are not streamed
  yet.
- **The stock scenes start with fingers intersecting the shelf.** In every
  bundled scene, MABEL's index or thumb links already penetrate the shelf at
  reset. Teleop's default `--spawn-backoff` avoids this; the scenes still need
  a proper fix.
- **Head aiming from the image is approximate.** It uses a small-angle model
  that ignores the camera's offset from the neck axes; a second click refines
  the aim. Neck roll is not exposed.
- **Wrist servos are weak.** The upstream wrist actuators have `kp = 1`, so
  wrist angle sags under load or contact beyond what the bounded feedforward
  can correct. Status shows measured versus nominal values, and the model is
  not modified.
- **Reach is limited because hand orientation is held.** Raising or retracting
  a hand by about 10 cm from Ready while holding its orientation can exceed
  the joint limits. The target turns red; rotating the target extends reach.
- **IK is local and has no collision checks.** Teleop applies the closest
  bounded IK solution and reports the residual. Physics resolves contacts. This
  is never evidence of a grasp or placement.
- Twin textures are not shown (colours come from MuJoCo materials).
