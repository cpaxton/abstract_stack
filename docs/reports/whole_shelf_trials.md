# Stocking insertion development trials

These are successive development trials with different policies or initial conditions. They are not a frozen-policy multi-scene success-rate study. Full videos and traces remain under ignored `outputs/`; the companion JSON pins sources and artifacts.

Every row declares its scheduled carton count. Prefilled cartons earn no pick/place/retreat credit. All products are monitored at each physics step for transient tips and drops. A successful episode requires all scheduled stages and its configured final stability/packing checks.

| Trial | Total free mm | Prefilled | Clean | Pick/place/final/retreat | Outcome |
|---|---:|---:|---|---|---|
| tight-four-14 | spacious walls | 0 | no | 1/1/0/0 / 4 scheduled | finger_push: contact dwell timeout |
| tight-ten-09 | 10 | 0 | no | 1/1/1/1 / 10 scheduled | pregrasp: bounded staging complete; predicted remaining miss 0.306 m; replan required before grasping |
| tight-ten-10 | 10 | 0 | no | 2/1/1/1 / 10 scheduled | transfer: IK prefilter found no acceptable pose/path; predicted miss 0.054 m; replan required |

## Saved-state development probes

Every continuation earns zero fresh task credit. A verified retreat here is an isolated skill result, not a whole-scene success.

| Continuation | Release/retreat verified | Final yaw deg | Outcome |
|---|---|---:|---|
| four-first-replan-01 | False | -0.41672 | finger_push: joint-limited push needs a new base pose |
| four-first-replan-02 | False | -0.00012 | contact_refine: bounded release-goal search exhausted |
| ten-edge-transfer-01 | False | 2.45278 | lateral_approach: Cartesian feedback route rejected |
| ten-edge-transfer-02 | False | -3.39053 | lateral_push: final corrected reference bound |
| ten-edge-transfer-03 | False | 6.51401 | lateral_push: projected reference outside bounds |
| ten-edge-transfer-04 | False | 4.51218 | lateral_push: projected reference outside bounds |
| ten-edge-transfer-05 | False | 16.74700 | lateral_push: final corrected reference bound |
| ten-edge-transfer-06 | False | 19.74836 | lateral_push: final corrected reference bound |
| ten-edge-transfer-07 | False | -0.15955 | lateral_push: displacement/support dwell timeout; replan |
| ten-edge-transfer-08 | False | 0.00717 | pusher_plan: bounded base/posture search exhausted |
| ten-edge-transfer-09 | False | unknown | unknown process |
| ten-edge-transfer-10 | False | 0.00717 | pusher_plan: bounded base/posture search exhausted |
| ten-edge-transfer-11 | True | -0.07586 | completed probe |
| ten-pusher-exit-01 | True | -0.15482 | completed probe |
| ten-pusher-exit-02 | False | unknown | unknown process |
| ten-pusher-exit-03 | False | -0.00113 | release_forecast: no verified bounded reference/withdrawal candidate |
| ten-pusher-exit-04 | False | -0.00113 | release_forecast: no verified bounded reference/withdrawal candidate |
| ten-pusher-exit-05 | False | -0.00113 | release_forecast: no verified bounded reference/withdrawal candidate |
| ten-pusher-exit-06 | False | -0.00113 | release_forecast: no verified bounded reference/withdrawal candidate |

See [the dense task guide](../dense_stocking.md) for reproduction and limits.
