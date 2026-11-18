# Engagement and final command ownership (#20)

`EngagementGate` runs on the same thread as `MujocoTeleopAdapter`. It owns the
Disconnected, Ready, Engaged, Paused and Fault policy, current calibration
artifact/generation, bounded control-event replay cache and the only final
complete59-channel command application. Retargeters own proposals; they do not
write the live simulator. The gate never changes live qpos, model limits, masses,
contacts or external forces. The owner calls `sim.step()` after each successful
gate tick, including during holds. An explicit reset is a separate operation.

## Default admission and open qualification

The gate defaults to head/arms/fingers mode with torso/lift held. Body commands
require explicit body-required mode and a later #19 mapping. Missing body/elbow
guidance is optional in the default mode; stale/missing head or either hand
pauses the whole human-command group. Body-required mode needs both chest and
hips valid, not merely one valid body root.

**Human engagement is disabled by default unless two providers are installed:**

- A callable composed geometry guard, such as #17's `check_composed`, checking
  the complete actual proposed finger/arm/head/body-hold vector after final rate
  correction. This is a sampled scratch prefilter, not a dynamics guarantee.
- Fresh independent `LifecycleEvidence` with
  `removal_detection_qualified=True`. This flag may be set only after the actual
  provider's remove/re-wear behavior is independently instrumented and verified.
  Raw `worn:true` or the old probe's `user_present` alone does not qualify it.
  Current physical removal qualification remains open; test fixtures explicitly
  use synthetic qualified evidence and are not a completed headset safety pass.

`VideoEvidence.verified` must originate from #21's session/generation/sent-pair
and clock validation. Preserve the original capture and feedback host times.
Only a complete submitted pair qualifies; a decoder callback does not. Optical
motion-to-photon time is not measured by this component. The integration owner
must supply this provenance; constructing a trusted evidence object from raw
unvalidated client values is outside the supported contract.

## API and data flow

```python
retargeter = ArmHeadRetargeter(config, expected_model_sha256=sim.describe()["model_xml_sha256"])
gate = EngagementGate(sim, settings=GateSettings(enabled_groups=("head", "arms", "fingers")))
gate.composed_guard = lambda snapshot, complete: retargeter.check_composed(
    snapshot, complete, body_hold=gate.body_hold_references()
)
# Transport callbacks publish detached data through OwnerMailbox; the owner:
mailbox.drain(gate, now=host_monotonic_seconds)
# Run only the selected retargeters, wrapping each channel group separately.
command = gate.tick(proposals, now=host_monotonic_seconds)
if command is not None:
    sim.step()
mailbox.publish_status(gate, now=host_monotonic_seconds)
```

`CommandProposal` carries session ID, calibration generation, the newest accepted
tracking sequence, mapped host observation time, validity and named targets.
Each selected group must own exactly its declared channels. Split #17's19-entry
map into arms16/head3; #18 owns fingers32. Full base references are reserved:
reset-measured steering holds and zero drive speeds. Lift remains one native
`Total Lift` tendon-length command, not an arbitrary slide coordinate.

The gate compiles a private descriptor model and verifies generated XML hash and
actuator order against the adapter. Final native command bounds are checked
before and after rate correction, then complete geometry is checked, then one
atomic adapter apply occurs. Only one write is allowed per advancing physics
control step. External ctrl writes, unexpected resets, nonfinite physics,
unusable geometry guards and infeasible rate transitions fault before a new
unvalidated write. Retargeter filter history must be reset on gate rejection,
pause, session/generation changes or realignment. Pass actual applied references
back to the finger mapper when final rate correction changes its proposal.

## Lifecycle and clocks

The owner accepts strict canonical v1 tracking and copies it. App observations
are mapped with #15's verified host-minus-client estimate. Default maximum input
age is150 ms including uncertainty; clock uncertainty above25 ms or mapping age
above10 s is unusable. SDK hand sample timestamps must be positive and advance a
high-water mark; duplicate/regressed stamps cannot renew hand observation age.
Head predictions/body state times are not sensor freshness measurements.

Input, independent lifecycle and original video capture/feedback times are
rechecked on every tick, even with no new packets. New independent lifecycle
loss can pause a frozen pose sequence without refreshing pose age. Video capture
or feedback older than300 ms blocks motion; repeated feedback cannot rejuvenate
an old image. Disconnected/old-generation messages cannot silently establish new
alignment or replay reset.

`calibrate` is allowed only Ready/Paused with usable required evidence and
caller-supplied validated neutral poses/robot references. It advances generation,
clears prior observations and leaves Ready; new-generation observations and a
deliberate `engage` are required. A pause clears the artifact and requires a new
neutral and deliberate `resume` if motion was previously engaged. Restored input
never automatically resumes. Recenter invalidates and advances generation even
without an artifact. Pause/recenter accept obsolete incoming generations; other
actions require the current generation. Identical event retries return cached
acks; changed/reordered/reused IDs reject. The cache holds64 entries, event trace128
and retired session replay protection10,000 UUIDs. A fault needs an explicit
finite-state reset path; a tracking refresh cannot clear it.

## Holds and failure behavior

On pause the owner latches measured native position-servo actuator lengths,
reserves base references, and transitions command references with velocity and
acceleration bounds. Defaults:1 rad/s and5 rad/s²; Lift0.05 m/s and0.25 m/s².
Bounds use native simulation control dt. Braking caps reserve distance to native
ctrl limits; an empty rate/braking interval faults. The latched body hold can be
supplied to #17's complete-command guard for bounded transition admission.

Measured posture can be outside native command ranges. In that case, or after
other faults, the gate retains the last validated command reference and reports
Fault; it never clips a measured posture into a falsely certified hold or widens
limits. Fault retention is an emergency reference freeze, not a smooth velocity
transition, zero-effort stop or object-retention guarantee. Failed applies return
None so the owner can diagnose/stop stepping rather than continue new retargeting.

Host-side pause is always available. Reset checks finite native state/commands,
uses the adapter's explicit reset, discards queued alignment, and requires new
calibration. Nonfinite physics requires host diagnostics/restart; reset does not
hide it. No loaded-hold/grasp/lift/release acceptance is claimed.

## Bounded networking integration

`OwnerMailbox` offers a reliable capacity32 command queue (configurable1–64), one
newest detached input, and urgent disconnect/pause flags. Received inactive
lifecycle events are coalesced separately by session. The owner applies a loss
before consuming replacement input, so recovery cannot erase it or reuse the
old alignment. Queue overflow requests host pause. Cancelled pending commands do
not execute. Control decisions use the actual owner-drain time, not old enqueue
time. Calibration callbacks run on that owner. A 200 ms control RPC timeout or a
failed recenter RPC retires the session and closes its connection. Reconnect
with a fresh session and alignment. Drain at most8 reliable requests before the
control tick to avoid unbounded network work.

`GateControlServer` extends #15 without modifying its files. It publishes the
actual owner status rather than the base server's fixed Paused placeholder,
synchronizes session generations, and closes a connection if the owner status is
older than150 ms. Its observation provider must supply independent lifecycle
and verified #21 video evidence without reading live MuJoCo. A second control
session cannot take over an existing gate. Full #24 client/video/retargeter
integration remains a separate step; this component is not an APK or UI.

## Verification and native evidence

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_quest_engagement.py
MABEL_ROOT=/path/to/open_mabel PYTHONPATH=src python scripts/probe_quest_engagement.py
```

The policy regressions use clearly labelled detached servo fixtures for state
and limiter tests, plus actual native physics for the hold counterexample. An
actual loopback WebSocket test exercises clock/tracking, owner-thread calibration,
engage/paused acknowledgements and authoritative status. It does not report a
fixture as native physical execution. Optional composed tests with #17/#18 run
when both independent modules are available.

The native unloaded probe schedules100 ticks per case, seed0, default native
robot, no products, timestep0.002/frame_skip10. An explicit shelf-clearance scene
(front_x−1.4) completed100 Paused hold ticks with about0.160 mm base displacement;
zero wheel commands do not weld the base. The default shelf fixture uses #14's
ctrl+step measured-reference preparation and faults at its first gate tick:
Right Index Abd measured−1.087589 rad is outside native[−1.04577,0.24577]. It keeps
validated references; this is retained failure evidence, not a hold success.
Full traces/settings/model/source hashes are under ignored outputs. No object
was held and no physical Quest lifecycle qualification is implied.

## Stack and ownership

This task owns session.py/session_bridge.py, its tests/probe and this document.
#17 owns retarget_arms; #18 owns retarget_fingers. No shared simulator, transport,
contract, package or Unity setting is overwritten. The common development base
`codex/quest-control-base` contains provider PRs#26/#27/#28/#32–#35, including the
latest #27 bind-validation fix. The three child diffs remain disjoint. After
reviewed, authorized provider squash merges, rebase only each task's commits onto
actual main and retarget. No main/PR merge is performed here. Keep control PRs
draft while physical input/lifecycle, loaded-hold and full-stack gates remain open.
