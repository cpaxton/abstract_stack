# Sustained supported contact and squaring

The new check distinguishes first backstop contact from the full supported,
squared-contact dwell. Three bounded continuations preserve native force and
stability guards, but **none passes the squaring gate**. The best final yaw is
2.75 degrees; the retained source controller requires at most 0.15 degrees.
No release, retreat, complete stock or new episode success is claimed.

These are development continuations of the same failed ten-carton v17 episode
used in the [initial lookahead study](compliant_lookahead_development.md).
Every carton is monitored at every 2 ms native step. The native MABEL model,
inertias, gains, actuator limits, contacts and 25 N abort bounds remain unchanged.
There are no live pose writes, attachments or external-force actions.

## Controller and measurement changes

`BackstopDwell` admits only consecutive native physics samples. A missing,
repeated or reordered sample fails closed. At every sample, the original source
thresholds require back normal force >=0.4 N, hand and shelf support >0.2 N,
upright tilt <=5 degrees and absolute yaw <=0.15 degrees for 0.12 s. A contact
break or orientation violation resets the dwell. This is separate from final
position, release, retreat and the task evaluator.

The first probe keeps the original advance-only ranking. The second switches
to absolute-yaw ranking once the carton braces against the backstop, with an
off-center lateral target using the source controller's yaw-lever gain. Both
screening and fully guarded selection use the same objective. Unchecked
forecasts remain ineligible; the objective changes no force or collision gate.

The third probe reuses the dense controller's bounded yaw integral and angular
velocity damping. It uses the normal-force-weighted measured hand/carton contact
point for lateral feedback, with measured tool position as a fallback. Object
pose and contact identity are explicitly privileged teacher inputs, separate
from the student sensor stream. Desired contact lever stays within +/-14 mm;
all original nominal/applied motion, rate, range and coupled finger checks apply
after corrections. The proposal's wrench remains frozen over each 0.4 s
forecast, followed by measured-state replanning every 100 ms.

## All scheduled probes

| Probe | Scheduled/executed decisions | Simulated time | Carton advance | Final yaw | Peak robot/product | Parent wall time |
| --- | --- | --- | --- | --- | --- | --- |
| `dwell-v1` | 40/40 | 4.0 s | 9.92 mm | 4.43 deg | 2.33 N | 50.43 s |
| `square-v1` | 40/40 | 4.0 s | 10.16 mm | 3.92 deg | 3.00 N | 50.03 s |
| `square-v2-integral` | 80/80 | 8.0 s | 10.68 mm | 2.75 deg | 4.70 N | 82.44 s |

All three processes exited 0 within their declared 90/90/150-second budgets.
That denotes a completed forecast/continuation process, not a passed task.
All three have zero fresh task credit and unverified contact dwell because of
yaw. No transient tip/drop, force abort, non-grasping-arm contact, robot/fixture
contact or carton-pair load was recorded. The third probe reached 2.96 N peak
backstop force and 4.44 mm peak measured-base nominal-FK execution error.
Predicted IK residuals remain separate from measured execution tracking.

The third probe ends with only the held-target candidate generated: every
moving proposal is rejected by an existing nominal-motion/geometry or
projected-applied-reference bound. This local proposal failure does not establish
global unreachability. Extending duration alone does not solve the problem.
The controller needs a checked unload/backoff and different contact lever or
posture, followed by replanning; continuing to press at a bound is insufficient.
No such recovery was executed here.

The first two bundles report `failure=null` for physical execution while their
explicit dwell field is false. The current runner additionally records
`backstop_dwell_decision_budget_exhausted`, as seen in the third bundle. This
report retains their failed stage outcomes without rewriting earlier artifacts.
New nominal-rejection diagnostics capture the returned IK/limit report and
linear/angular bounds for future probes; they were added after these runs.

## Reproduction and provenance

Full bundles remain in `outputs/compliant-lookahead-v2/`. The
[JSON ledger](sustained_contact_development.json) retains settings, process
outcomes, source hashes, selected measurements and final rejection reasons for
all three scheduled probes. Source revision was
`975c950ec646c740e643907a90ad1866d9ff3adf` plus each bundle's copied uncommitted
sources, identified by SHA256. Native MABEL revision is
`d0181f04025c06e1790b7ab41fe44d602cc2ee5b`; MuJoCo 3.17.0 and NumPy 1.26.4.
All start from the same integration-state hash
`d2acf8ae96cd6ffc25f67a88649729511c8f53f52028c6b8940e7d41ffd481b2`.

With the retained source episode available, use a fresh output directory:

```bash
.venv/bin/python scripts/probe_compliant_lookahead.py \
  --episode outputs/contact-rich-replanning-v1/mobile-table-ten-v17/episode \
  --output outputs/my-fresh-squaring-probe \
  --candidates 8 --shortlist 4 --horizon 0.4 \
  --execute-prefix 0.1 --require-dwell --square-feedback --replans 80 \
  --generation-seconds 10 --wall-seconds 150
```

Historical results require their exact frozen bundle sources; the current
command records current sources. Native process instability remains an open
acceptance gate despite the passing publication suites. The next task gate is
successful squaring with support preserved, then unloading and clean retreat,
then integration into a fresh full pick/carry/stock sequence. Planning remains
slower than real time. No whole-shelf success rate is inferred from this study.

## Publication stability evidence

The first required pre-push at `3a0dcab` collected 310 tests, passed seven and
aborted with signal 6 during `MjModel.from_xml_string` in a base-control fixture.
Git exited 1 and no branch was published. The full log/process outcome remains
in `outputs/pr-publication-squaring-v1/`. This occurred before execution of the
new contact gate/search, but the Python frame does not identify a native cause.
It is retained independently of earlier passing full suites and subsequent
allocator diagnostics. The source commit remains in this branch's history.

The subsequent full hook at `d886e1a` used `PYTHONMALLOC=debug` and also aborted
with signal 6 during fresh model compilation, in a released-contact fixture.
Its log and effective allocator/source flags remain in
`outputs/pr-publication-squaring-v2-debug/`. Neither attempt produced terminal
test XML or published this branch. No hook was bypassed and no physics or
acceptance criterion was changed. The allocator diagnostic did not establish a
workaround; the native cause remains unknown. Focused checks, package build and
the completed development continuations do not replace these failed full checks.
