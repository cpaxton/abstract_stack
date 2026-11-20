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
640x480 per eye), `--video-bitrate` (H.264 ceiling, default 1500 kbit/s) and
`--video-quality` (JPEG quality, default 80). `?video=jpeg` in the address
forces the JPEG fallback.

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
measured palm. When the requested orientation is out of reach, the hand keeps
the palm on target and twists only as much as it must; the arm card shows
"Twisted N°". The target turns red when the point itself is out of reach, or
when the collision guard stops the arm short ("Blocked: shelf", "table",
"own body"). Leaving the tab pauses immediately.

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
  robot-gravity feedforward) → `validate_targets`. IK
  (`PoseIK.solve_position_first`) treats the palm point as the primary task and
  recovers orientation in its null space, so a reachable pose is solved exactly
  and an unreachable one keeps the palm on target. A step that moves the palm
  away is undone and retried with less orientation gain, joints pinned at a
  limit are dropped, and a position-only restart escapes limit traps. IK
  warm-starts from the previous solution and keeps refining until it converges
  or stops improving. Only the IK result is cached. Every tick, a kinematic
  collision guard (scratch data) samples each arm's joint path from its
  measured pose to the IK result, about every 0.025 rad, against fixtures and
  MABEL's own body. It then samples both arms' paths together, so two arms that
  are each clear cannot be commanded into each other. Products are excluded so
  the hands can touch and grasp them, and contacts already present are allowed.
  Each arm moves only to the farthest collision-free point on its path. Because
  the check reruns against the current scene, a block clears as soon as driving
  or the other arm makes room. Lift is the
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
  unsubscribes when no camera view is visible. Head-camera rendering is
  triangle-bound, so the render thread uses its own copy of the model in which
  meshes used only by the base (swerve modules, wheels, chassis, lift mount)
  and the head shell are simplified to about 8k faces each. The arms, hands
  and torso stay at full resolution. The copy is pixel-identical at Ready with
  the head fully down, and physics never sees it. Measured on an
  Apple-silicon Mac: render 31 ms per pair instead of 43 ms, 20 pairs/s at
  640×480 per eye, about 37 KB per pair (6 Mbit/s), physics real-time factor
  1.0. The client decodes off the main thread (`createImageBitmap`), keeps
  only the newest pair, swaps both eyes together, and draws each pair as soon
  as it decodes instead of waiting for the next animation frame.
- **Client performance.** 50 Hz poses go into an adaptive jitter buffer. It
  plays back one frame period plus the 95th-percentile arrival lateness
  (+4 ms) behind the fastest observed arrival, and interpolates at display
  rate. That is about 28 ms on a LAN, and it grows on its own with jitter.
  Commands go out once per display frame, just after that frame's input
  handling (at most 125/s, with a 60 Hz timer fallback). High-rate data never
  touches React state; panels re-render at most 10 Hz.

### Efficient streams for slow links

Everything adapts to the operator's link, so a phone on mobile data stays
responsive instead of falling seconds behind.

- **H.264 video** (`video.py`). Both eyes are encoded side by side in one
  stream with libx264, the same on macOS, Linux and Windows (bundled in the
  PyAV wheels of the `teleop` extra): constrained baseline, zero latency, no
  B-frames. Browsers decode it with WebCodecs, which needs a secure context
  (HTTPS, or localhost). It's about 1.3 Mbit/s for motion that costs about
  6.7 Mbit/s as JPEG, and encodes in under 2 ms. Viewers start on a keyframe,
  and a decoder error asks for a new one. Browsers without WebCodecs H.264 get
  the JPEG stream. Apple's hardware encoder was measured and rejected: it
  buffered four frames (+200 ms).
- **Compact poses.** Clients ask for `WTS2` frames: int16 positions
  (1/4096 m) and quaternions, half the size of float32 `WTS1`
  (0.92 → 0.46 Mbit/s at 50 Hz).
- **Flow control.** The browser acks each pose and video frame on arrival. The
  host keeps at most about one round trip plus slack of each stream in flight
  (90th-percentile round trip, capped at 1 s). Beyond that, poses skip to the
  newest frame and H.264 resumes at a keyframe, so buffers along the path never
  fill with stale frames. Frames left unacked for 1.5 s count as lost.
- **Link controller** (`link.py`). Four times a second the operator's browser
  reports queueing delay: one-way pose and video delay above their recent
  minimum, on the synced clock. Windows where the page itself stalled are
  skipped, because a slow device must not look like a slow link. Above 80 ms
  the host backs off, in order: H.264 bitrate (down to 250 kbit/s), pose and
  status rate (25, then about 17 Hz), and video frame rate (10, then about
  7 fps). Below 25 ms it recovers in reverse. Status shows it as `link`.

Measured with Chromium's network emulation (40 ms added latency; rates are
what the emulator actually delivered), host and browser on one Mac:

| Delivered link | Before: poses / video (p50) | After: poses / video (p50, p95) |
|---|---|---|
| unthrottled | 27 / 35 ms | 27 / 34, 35 ms |
| ≈1.1 Mbit/s | 51 ms / 7.1 s | 49 / 48, 99 ms |
| ≈0.33 Mbit/s | 5.9 s / 10.3 s | 131 / 88, 232 ms (250 kbit/s, ~7 fps video) |
| ≈0.27 Mbit/s | — | too little: the stale-view guard pauses |

### Latency

The client syncs its clock to the host's `time.monotonic()` (NTP-style, using
the lowest round trip of the last 16 pings: a burst of 5 at connect, then 1/s).
It then measures each stage on one clock:

| Stage | What it covers | Localhost before | Localhost now |
|---|---|---|---|
| Input → send | intent update to command on the wire | 17 ms (30 Hz timer, mean) | 0.3 ms |
| Send → applied | network + wait for the next 50 Hz physics tick | 11 ms | 5–16 ms |
| 3D view | host physics to pose drawn (network + jitter buffer) | 38 ms | 28 ms |
| Camera | capture to frame drawn (render, encode, network, decode) | 49 ms | 35 ms |

Values are medians on an Apple-silicon Mac with headless Chromium on GPU. "Drawn"
means the canvas was updated; the display adds up to one refresh. Send → applied
now depends on how the browser's frame timing happens to line up with the host's
50 Hz tick (5 ms in one run, 16 ms in another; 0–20 ms is the possible range).
Input → applied fell in every run (p95 about 35 → 18 ms or less). To reproduce
against a running host:

```bash
cd web && node scripts/measure-latency.mjs "http://127.0.0.1:8080/?token=<token>" 20
```

The connection chip shows the latency of the view you steer by ("3D 28 ms" or
"Video 35 ms"). Its tooltip and the Diagnostics card break it down by stage.

**Stale views pause.** Commands keep flowing when only the picture freezes, so
the host's input timeout would never fire. The client therefore watches what
you actually see in the view you steer by: the pose the 3D view last drew
(which plays back behind the newest arrival), stale after 250 ms; or the camera
frame last drawn, stale after 300 ms. A view that hasn't drawn anything yet
counts as unavailable. Engage checks the selected view at the moment you press
it, so switching to a camera with no frame yet and engaging straight away is
refused. An engaged session pauses itself and names the reason ("Paused: camera
video is 420 ms old"). Before an automatic pause, staleness has to persist for
150 ms, and the page's own stalls (a long frame on a slow device) are
discounted, so a slow phone does not pause spuriously.

WebSocket compression (permessage-deflate) is off by default, since frames are
JPEG and float32 and gain little. Pass `--ws-deflate` to turn it on.

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
2. State socket `/teleop/state`: `state_bind {session_id, format?}`, then binary
   frames. Each frame is a 28-byte little-endian header (`"WTS1"`, tick u32,
   sim_time f64, host_time f64, nbody u16, flags u16), then float32
   `xpos[nbody·3]` and `xquat[nbody·4]` (wxyz, world frame). With
   `format: "wts2"` the magic is `"WTS2"` and the values are int16: positions in
   1/4096 m, quaternion components ×32767. The client sends
   `state_ack {tick}` for each frame it receives.
3. Video socket `/teleop/video`: `video_bind {session_id, codecs?}`, where
   `codecs` is the client's preference (`["h264", "jpeg"]`), then binary
   messages. `QTP1` is JPEG pairs. `QTH1` is H.264: ASCII `QTH1`, big-endian
   uint32 metadata and payload lengths, JSON metadata (`pair_id`,
   `capture_host_monotonic_s`, `keyframe`, `layout: "side_by_side"`,
   `bitrate_kbps`, …), then one Annex-B access unit, with SPS/PPS in-band on
   keyframes. The client sends `video_ack {pair_id}` per message, and
   `video_keyframe` after a decoder error. `hello_ack.video` lists `codecs` and
   describes the cameras, size, rate and field of view, or is `null` with
   `--no-video`.
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
6. `ping {t0}` → `pong {t0, t1}`: round trip, and clock sync (`t1` is the
   host's `time.monotonic()`).
7. `status` (10 Hz) includes `command_timing {sequence, sent_ms,
   received_host_s, applied_host_s}` for the newest applied command, so the
   client can measure send → applied. The host keys it by (session, sequence),
   clears it when the operator changes, and sends it only to the session that
   sent the command. Every other session gets `null`, because `sent_ms` is on the
   sender's browser clock.
8. `link_feedback {queue_ms}` (operator, 4 Hz): queueing delay measured by the
   browser; the host adapts the streams (status `link`).

**Target epoch.** Whenever the host moves arm references on its own (engage,
ready, reset), it increments `target_epoch` and publishes `arm_targets`. The
client rebases its targets on those, and the host ignores arm intents from an
older epoch. That way a stale ghost can never yank the robot back.

**State machine.** Disconnected → Ready (operator connected) → Engaged
(deliberate engage) ⇄ Paused (operator pause, tab hidden, input older than
`--input-timeout`, the view being steered by going stale, episode time limit). Fault covers non-finite physics or an
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
# The host needs only the teleop extra; the WebSocket tests also need httpx from the
# ci group (they skip with this hint otherwise). The repo pins Python 3.11
# (.python-version): a transitive open3d dependency has no 3.12 wheel.
uv sync --extra teleop --group ci
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

The browser e2e tests exercise connect, keyboard jog, pause on input loss,
stereo video, driving, click-to-look, the body rails, the first-run guide and
the touch layout. Headless Chromium on Linux rasterises WebGL on the CPU (about
1–10 fps), which stalls the main thread between frames. The tests therefore
use `?quality=low`, re-engage right before each action, and run the host with
`--input-timeout 2.0`. Input-loss pausing has its own e2e test and is also
covered natively.

On macOS the tests use the GPU. `E2E_GPU=0` forces CPU rendering, and
`E2E_CPU_THROTTLE=4` emulates a slower runner, for example
`E2E_GPU=0 E2E_CPU_THROTTLE=4 npm run e2e`.

## Limitations

- **Latency is measured to the canvas, not the eye.** Every stage is on one
  synced clock (accurate to ± half the best round trip), but the display's
  scan-out, up to one refresh, is not included. Stale-view pausing is enforced
  by the client; the host cannot check what the operator actually sees, and
  relies on its own input timeout.
- **Video rides TCP (WebSocket).** Flow control and the link controller keep
  latency bounded and degrade gracefully, but a lost packet still delays the
  frames behind it, where WebRTC over UDP would not. One shared encoder and
  rate follow the operator's link, so a spectator on a slower link gets its
  frames skipped. Wrist cameras are not streamed yet.
- **Link numbers come from browser emulation** on one machine, not a real
  mobile network.
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
- **Reach.** On a 385-point grid around the left Ready palm, with the Ready
  orientation held, 70% of targets are reached at full pose and 15% more by
  relaxing orientation (55% before). The rest are kinematically out of reach,
  or blocked by the guard (mostly the other arm or the torso). Lift is the
  operator's channel and is not used to extend reach.
- **The collision guard is kinematic and local.** It samples the straight
  joint-space path from the measured pose, so obstacles thinner than the
  sample spacing can slip between samples. It does not model the controller's
  exact rate-limited path or plan around obstacles: a target behind an obstacle
  leaves the arm stopped at it. Products are never obstacles.
  Physics still resolves every contact. Nothing here is evidence of a grasp or
  placement.
- Twin textures are not shown (colours come from MuJoCo materials).
