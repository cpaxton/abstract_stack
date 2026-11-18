# Web teleoperation (browser, phone, tablet)

Teleoperate MABEL in `ShelfStockingEnv` from a browser. The operator sees an
interpolated 3D twin of the simulated robot and scene, drags or jogs a target
for each palm, and the host turns those targets into bounded actuator commands.
Desktop uses mouse and keyboard; phones and tablets get on-screen joysticks.
The same client is meant to be wrapped later as an iOS/Android app (Capacitor)
and extended with WebXR.

Status: simulation only. Robot-camera video, session recording and WebXR are
not implemented yet (see [Limitations](#limitations)).

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
# open http://<computer-ip>:8080/?token=<token> on the device
```

The token is generated per run unless `--token` is given. The client stores it
for the tab and strips it from the address bar. `--insecure-no-token` disables
it on a trusted development machine. A second browser joins as a view-only
spectator while an operator holds control; `?role=spectator` requests that
explicitly. `?quality=low` forces the light render path, which the client also
switches to automatically when the frame rate drops.

Useful options: `--scene`, `--seed`, `--time-limit` (default 3600 simulated
seconds), `--input-timeout` (default 0.15 s), `--face-budget` (twin mesh detail).

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
| Switch arm | <kbd>Tab</kbd> | arm chip |
| Head, lift | arrows, <kbd>PgUp</kbd>/<kbd>PgDn</kbd>, sliders | panel sliders |
| Fine motion | hold <kbd>Shift</kbd> | **Fine** toggle |
| Ready pose | <kbd>R</kbd> | top-bar button |
| Camera presets / help | <kbd>1</kbd>–<kbd>4</kbd> / <kbd>?</kbd> | panel |

Start with **Ready pose**. The reset posture has both arms straight ahead, an
elbow singularity where Cartesian control responds poorly. Ready bends the
elbows and brings the palms about 0.4 m in front of the chassis. Each glowing
target shows where that palm is commanded to go. A tether connects it to the
measured palm, and the target turns red when IK cannot reach it. Leaving the
tab pauses immediately.

## Architecture

```
Browser (web/)                                  Host (src/abstract_stack/web_teleop/)
 TeleopClient ── WS /teleop/control (JSON) ──►  server.py  Starlette/uvicorn, auth, fan-out
   commands 30 Hz, control, ping   ◄── status 10 Hz, acks, pong
 PoseBuffer  ◄── WS /teleop/state (binary WTS1, 50 Hz body poses)
 robotModel  ◄── GET /api/model (+ model.bin)   geometry.py  visual geoms from the live MjModel
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
  thumb preshape. Head and torso are clamped and rate-limited. The base gets
  stationary targets.
- **Nominal, applied and measured are kept separate.** Status reports IK
  position error (m) separately from orientation error (rad), the measured
  palm pose, and which joints were rate- or target-limited.
- **Delivery is latest-wins.** Each socket's sender transmits the newest
  frame/status when it is ready, so a slow peer skips frames instead of
  queueing them or slowing physics.
- **The twin is the real model.** `geometry.py` exports visual geom groups 0,
  2 and 5 (scene, robot visuals, house cutaway walls) from the compiled
  `MjModel`. Meshes are simplified to `--face-budget` faces each (about 210k
  faces / 1.8 MB gzipped for the stocking scene), cached by content hash, and
  served immutable. The simplification affects the preview only, never
  physics.
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
2. State socket: `state_bind {session_id}`, then binary frames. Each frame is
   a 28-byte little-endian header (`"WTS1"`, tick u32, sim_time f64,
   host_time f64, nbody u16, flags u16), then float32 `xpos[nbody·3]` and
   `xquat[nbody·4]` (wxyz, world frame).
3. `command`: absolute intents with `calibration_generation`, `target_epoch`,
   `arms.{left,right}` (palm control-point `position_m` and `quaternion_wxyz`
   in the `mabel_chassis` frame, plus `grip` ∈ [0,1], or null to hold), and
   `head`, `lift_m`, `torso_rad` (null holds). Absolute targets make every
   message self-contained, so a lost message never accumulates drift.
4. `control {event_id, calibration_generation, action}` with actions
   `engage | pause | reset | ready` → `control_ack`.
5. `ping {t0}` → `pong {t0, t1}` for round-trip display.

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
- the WebSocket handshake, token check and state stream.

The browser e2e test exercises connect, keyboard jog, pause on input loss, and
the touch layout. Headless Chromium rasterises WebGL on the CPU (about
5–10 fps), so the e2e host runs with `--input-timeout 0.8`.

## Limitations

- **No robot-camera video yet.** The operator sees the client-side twin only.
  WebRTC streaming of `zed_left`/`zed_right` and the wrist cameras is the next
  step.
- **The stock scenes start with fingers intersecting the shelf.** In every
  bundled scene, MABEL's index or thumb links already penetrate the shelf at
  reset (`stocking`, `tight`, `house`: index fingers; `pick_place`: right
  thumb). The left index can snag while the arms move. The physics tests avoid
  this by moving fixtures away; teleop uses the scenes unchanged.
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
- The twin's meshes are simplified, and textures are not shown.
