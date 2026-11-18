# Stocking insertion development trials

These are successive development trials with different policies or initial conditions. They are not a frozen-policy multi-scene success-rate study. Full videos and traces remain under ignored `outputs/`; the companion JSON pins sources and artifacts.

Every row declares its scheduled carton count. Prefilled cartons earn no pick/place/retreat credit. All products are monitored at each physics step for transient tips and drops. A successful episode requires all scheduled stages and its configured final stability/packing checks.

| Trial | Total free mm | Prefilled | Clean | Pick/place/final/retreat | Outcome |
|---|---:|---:|---|---|---|
| tight-four-14 | spacious walls | 0 | no | 1/1/0/0 / 4 scheduled | finger_push: contact dwell timeout |
| tight-ten-09 | 10 | 0 | no | 1/1/1/1 / 10 scheduled | pregrasp: bounded staging complete; predicted remaining miss 0.306 m; replan required before grasping |
| tight-ten-10 | 10 | 0 | no | 2/1/1/1 / 10 scheduled | transfer: IK prefilter found no acceptable pose/path; predicted miss 0.054 m; replan required |
| tight-ten-11 | 10 | 0 | no | 2/2/2/2 / 10 scheduled | base_reposition: measured pose/stop dwell timeout |
| tight-ten-12 | 10 | 0 | no | 1/1/0/0 / 10 scheduled | release_forecast: no verified bounded reference/withdrawal candidate |
| tight-ten-13 | 10 | 0 | no | 2/2/1/1 / 10 scheduled | pusher_plan: bounded base/posture search exhausted |
| tight-ten-14 | 10 | 0 | no | 1/1/0/0 / 10 scheduled | finger_push: contact dwell timeout |
| tight-ten-15 | 10 | 0 | no | 1/1/0/0 / 10 scheduled | lateral_push: final corrected reference bound |
| tight-ten-16 | 10 | 0 | no | 1/1/0/0 / 10 scheduled | lateral_push: final corrected reference bound |
| tight-ten-17 | 10 | 0 | no | 1/1/0/0 / 10 scheduled | release_front: no unloaded support after opening |
| tight-ten-18 | 10 | 0 | no | 2/1/1/1 / 10 scheduled | transfer: IK prefilter found no acceptable pose/path; predicted miss 0.038 m; replan required |
| tight-ten-19 | 10 | 0 | no | 2/2/1/1 / 10 scheduled | payload_load: compensated arm/finger geometry rejected |
| tight-ten-20 | 10 | 0 | no | 2/2/1/1 / 10 scheduled | finger_push: contact dwell timeout |
| tight-ten-21 | 10 | 0 | no | 3/3/2/2 / 10 scheduled | release_forecast: no verified bounded reference/withdrawal candidate |

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
| ten-inventory-base-01 | True | 0.02677 | completed probe |
| ten-finger-load-insert-01 | True | 0.04381 | completed probe |
| ten-finger-load-insert-02 | True | -0.02650 | completed probe |
| ten-finger-load-edge-01 | False | 0.11691 | contact_refine: bounded release-goal search exhausted |
| ten-finger-load-edge-02 | False | 0.01766 | release_forecast: no verified bounded reference/withdrawal candidate |
| ten-finger-load-edge-03 | False | 0.01766 | release_forecast: no verified bounded reference/withdrawal candidate |
| ten-finger-load-edge-04 | True | 0.01963 | completed probe |
| ten-rigid-contact-push5-01 | False | 0.11795 | contact_refine: measured tool left fixed corridor |
| ten-rigid-contact-push5-02 | False | 0.04726 | contact_refine: bounded release-goal search exhausted |
| ten-rigid-contact-push5-03 | False | 0.08796 | contact_refine: no fresh guarded candidate |
| ten-rigid-contact-steering-01 | False | -0.45474 | contact_refine: measured tool left fixed corridor |
| ten-rigid-contact-abduction-01 | False | 0.43336 | contact_refine: measured tool left fixed corridor |
| ten-rigid-contact-tilt10-01 | False | 0.74223 | contact_refine: bounded release-goal search exhausted |
| ten-rigid-contact-single-01 | False | -0.66196 | contact_refine: bounded release-goal search exhausted |
| ten-wall-first-projection-01 | False | -1.58572 | lateral_push: displacement/support dwell timeout; replan |
| ten-wall-first-opening-01 | True | -0.02650 | completed probe |
| ten-loaded-route-01 | False | 5.02813 | pusher_plan: bounded base/posture search exhausted |
| ten-front-square-01 | False | 0.12211 | payload_load: compensated arm/finger geometry rejected |
| ten-front-square-02 | False | 0.07562 | payload_load: compensated arm/finger geometry rejected |
| ten-front-square-03 | False | 0.14900 | payload_load: compensated arm/finger geometry rejected |
| ten-front-square-04 | False | 0.14900 | pusher_plan: bounded base/posture search exhausted |
| ten-front-square-05 | False | 0.14900 | payload_load: compensated arm/finger geometry rejected |
| ten-front-square-06 | False | -4.28039 | payload_load: compensated arm/finger geometry rejected |
| ten-front-square-07 | False | -0.16069 | payload_load: compensated arm/finger geometry rejected |
| ten-front-square-08 | False | -0.85110 | payload_load: compensated arm/finger geometry rejected |
| ten-front-square-09 | True | -0.03930 | completed probe |
| ten-front-square-10 | True | 0.02814 | completed probe |
| ten-front-square-11 | True | 0.01006 | completed probe |
| ten-front-square-breakaway-01 | True | 0.02814 | completed probe |

See [the dense task guide](../dense_stocking.md) for reproduction and limits.
