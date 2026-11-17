# Quest control transport (#15)

This is a non-actuating WebSocket host and a Unity sender for the v1 contract in
[README.md](README.md). It deliberately rejects engage/resume/reset/calibrate
without an injected controller. It does not implement the #20 engagement gate or
claim physical teleoperation success.

Run from this checkout with its dependencies installed:

```sh
PYTHONPATH=src python scripts/probe_quest_transport.py --host 0.0.0.0 --port 8765
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_quest_transport.py
```

The standalone host exposes `/quest/control`; #21 attaches `/quest/video` through
`ControlServer(extra_routes={"/quest/video": handler})`. This local development
transport has no authentication/TLS configuration; bind to an intended development
interface. The default is loopback. The pinned `websockets` dependency uses the
[official asyncio server API](https://websockets.readthedocs.io/en/stable/reference/asyncio/server.html).
Python 3.10 uses 16.1.1; Python 3.11+ uses 17.4, matching supported interpreter
ranges without changing the simulator dependencies.

## Integration API

`await ControlServer(host, port, registry=SessionRegistry(), extra_routes=...,
on_control=..., on_video_status=..., hello_metadata=...).start()` returns the
server; `bound_port` supports ephemeral loopback tests, `await close()` closes all
peers. A separate50Hz status task reports Paused with an explicit reason, clock
estimate and observation ages; unknown simulator/command/error values are null. Handlers are async or synchronous callbacks:

- Extra route: `async handler(connection, registry)`. The connection is the
  library's `ServerConnection`. Only the route handler receives from its socket.
- `registry.get(session_id)` returns the current `Session` or None.
  `registry.bind_video(session_id, connection)` synchronously rejects orphan,
  duplicate or inactive binds and returns the session. `unbind_video` only clears
  the matching connection. The video handler owns bind JSON validation and ack.
- `registry.add_disconnect_callback(callback)` registers a callback accepting the
  session ID. Disconnect removes the session, clears observations, closes its
  `video_connection`, then invokes callbacks. Retired session UUIDs cannot be
  reused during this server lifetime (10,000 lifetime session ceiling).
- `on_control(session, message)` returns None to reject an action or an ack field
  dictionary. It is responsible for the future controller's policy. Event IDs
  advance; the last 64 identical retries return the cached ack. Changed payloads
  reusing an ID and expired/reordered IDs fail closed. The callback must not block
  the event loop; a future physics integration should enqueue reliable commands.
- `on_video_status(session, message)` receives strictly validated generic feedback.
  The video service additionally checks sequence, generation, sent-pair identity,
  presentation chronology and actual capture age.

Session public fields: `session_id`, `active`, `calibration_generation`,
`latest_tracking` (one newest validated message), `last_tracking_sequence`,
`video_connection`, `clock`. `await session.send({"type": ..., ...})` reliably
sends status independent of the pose slot, with a two-second send timeout.
`set_calibration_generation(n)` clears queued tracking, observation ages and SDK
sample high water marks. Recenter advances the generation before dispatching the
control callback, including when its incoming generation is obsolete. A reconnect
creates a new session with generation zero, no clock, no observations, no event
cache and no video binding. No motion is automatically engaged.

`session.clock` is None until a verified four-timestamp exchange, then has
`offset_s` (host minus client), `uncertainty_s`, and
`established_monotonic_s` (host). Map an app timestamp by adding `offset_s`.
Estimates expire after 10 seconds; uncertainty above 0.025 seconds is unusable.
Only the host's saved t0/t1/t2 exchange can establish an estimate. There are eight
pending exchanges, a two-second exchange lifetime and a one-second RTT ceiling.
`observation_ages(now)` reports mapped observation age plus uncertainty, never
sensor age. SDK sample timestamps are used only as hand high water marks: repeated
or regressed hand samples cannot refresh availability. All consumers must check
ages, generation, worn/focused flags and each component validity before use;
`latest_tracking` alone is not an engagement authorization. Stale or unsynchronized
input is never promoted as a fresh observation. Status, control and clock traffic
never pass through or get overwritten by that one-item pose slot.

## Unity integration

Add `QuestTeleop.Transport.QuestTransportClient` and
`MetaCanonicalTrackingSource` to a GameObject and assign the source's Client
reference. Set ControlUri to the host's reachable LAN address. The existing scene
is intentionally not rewritten by this PR. The source requires the existing
Meta Core 209 runtime configured for legacy OVR24 and body-tracking access.
OpenXR26 is rejected rather than mislabeled as OVR24. By default the source
requests Android body permission once, retries starting upper-body tracking after
a grant, and stops only its own tracking session on disable. Set
`ManageBodyTracking=false` when another component owns the SDK tracking lifecycle;
permission denial continues to produce unavailable body data.

The source retrieves one native SDK state per provider per Unity Update and uses
`Stopwatch` acquisition timestamps, separate from runtime sample/target times.
Head timestamps are prediction targets (`fresh:null`); body state targets carry
only diagnostic advancement. Hands preserve `SampleTimeStamp` and a monotonic
sample high water mark. Missing SDK/provider data has null positions/orientations
and explicit false validity, never identity/zero placeholders. Hand joint globals
are reconstructed from that same sampled state using SDK209's legacy hierarchy:
FlipX bind translations and local rotations, wrist right-multiplied by Y180,
FlipZ root and uniform hand scale. The SDK reports whole-hand tracking; individual
joint tracking/confidence fields stay null. Joint validity describes availability
of the composed pose, not independent optical evidence. Body18 selects the named
tracking-space joints and preserves their individual SDK validity/tracking flags.
Both head and body globals use FlipZ before the canonical reflection B=(z,-x,y).

`SubmitTracking(JObject canonical)` is also an injection seam for another provider.
It deep-copies data and preserves each source timestamp. `SendReliable(JObject)`
and the main-thread `MessageReceived` event connect status/control/video UI.
`MonotonicSeconds` is the common process clock for the viewer and sender. One
newest observation is sent at at most72 Hz; reliable queues are limited to64
messages. A stalled status consumer or reliable queue fails closed. Reconnect
uses exponential delay capped at4 seconds and a new UUID, clearing queued state.
Application suspend cancels the connection; reactivation reconnects and resyncs.
Recenter discards queued motion and suppresses new submissions until an ack.

`CanonicalReplaySource` accepts canonical JSONL, not RawTrackingRecorder files.
By default it preserves recorded source and observation timestamps so an old
recording becomes stale. Explicit `SyntheticRetiming` shifts application acquisition
timestamps by a single constant while preserving inter-source offsets; it does
not retime SDK timestamps or constitute live tracking evidence. Use a JSONL
conversion of the synthetic contract fixture for a network-only smoke test.

## Validation and remaining acceptance

Python tests exercise an actual loopback WebSocket session, clock exchange,
tracking/control interleave, session replay rejection, bound video disconnect,
strict skeleton/finite/quaternion/null validation, unknown fields, duplicate JSON
keys, bounded input and sample timestamp non-advancement. C# compilation and
physical Quest tests are recorded separately during integration; this transport
change alone does not establish full-device fresh hand/body delivery, recovery or
robot-control safety. Device validation must include absent/occluded hands,
permission denial, reconnect, recenter, suspend and slow receiver behavior.
