# Stocking insertion development trials

These are successive development trials with different policies or initial conditions. They are not a frozen-policy multi-scene success-rate study. Full videos and traces remain under ignored `outputs/`; the companion JSON pins sources and artifacts.

Every row declares its scheduled carton count. Prefilled cartons earn no pick/place/retreat credit. All products are monitored at each physics step for transient tips and drops. A successful episode requires all scheduled stages and its configured final stability/packing checks.

| Trial | Total free mm | Prefilled | Clean | Pick/place/final/retreat | Outcome |
|---|---:|---:|---|---|---|
| tight-four-05 | spacious walls | 0 | no | 1/0/0/0 / 4 scheduled | align_shelf: IK prefilter found no acceptable pose/path; predicted miss 0.080 m; replan required |
| tight-four-06 | spacious walls | 0 | no | 4/3/3/3 / 4 scheduled | base_reposition: final position guard |
| tight-four-07 | spacious walls | 0 | no | 0/0/0/0 / 4 scheduled | Process ended without a terminal summary; inspect run.log |
| tight-four-08 | spacious walls | 0 | no | 2/2/1/1 / 4 scheduled | release: withdrawal_released_squared_support_or_clearance_not_verified |
| tight-four-09 | spacious walls | 0 | no | 3/3/2/2 / 4 scheduled | stage_align: bounded pose-reference correction exhausted |
| tight-four-10 | spacious walls | 0 | no | 2/2/1/1 / 4 scheduled | finger_push: feedback path rejected before actuation |
| tight-four-11 | spacious walls | 0 | no | 1/1/0/0 / 4 scheduled | finger_push: contact dwell timeout |
| tight-four-12 | spacious walls | 0 | no | 1/1/0/0 / 4 scheduled | contact_refine: bounded release-goal search exhausted |
| tight-four-13 | spacious walls | 0 | no | 4/4/3/3 / 4 scheduled | finger_push: contact dwell timeout |
| tight-ten-05 | 10 | 0 | no | 1/1/0/0 / 10 scheduled | pusher_plan: no acceptable posture/path in bounded search |
| tight-ten-06 | 10 | 0 | no | 1/0/0/0 / 10 scheduled | base_reposition: measured pose/stop dwell timeout |
| tight-ten-07 | 10 | 0 | no | 1/1/0/0 / 10 scheduled | release_front: coupled stationary opening rejected |
| tight-ten-08 | 10 | 0 | no | 1/1/1/1 / 10 scheduled | retract: IK prefilter found no acceptable pose/path; predicted miss 0.000 m; replan required |

## Saved-state development probes

Every continuation earns zero fresh task credit. A verified retreat here is an isolated skill result, not a whole-scene success.

| Continuation | Release/retreat verified | Final yaw deg | Outcome |
|---|---|---:|---|
| four-center-pivot-01 | False | -6.09888 | stage_align: measured-pose feedback rejected before actuation |
| four-centered-insert-01 | False | 0.56302 | staging_clear: bounded staging complete; predicted remaining miss 0.003 m; replan required before grasping |
| four-fully-unloaded-01 | False | -0.24959 | release: withdrawal_released_squared_support_or_clearance_not_verified |
| four-last-insert-01 | False | -0.36899 | finger_push: feedback path rejected before actuation |
| four-loaded-base-01 | False | -4.27417 | stage_align: measured-pose feedback rejected before actuation |
| four-paired-insert-01 | False | 1.07788 | finger_push: carton tilting; replan required |
| four-paired-insert-02 | False | 3.09694 | finger_push: contact dwell timeout |
| four-paired-insert-03 | False | 44.02952 | finger_push: contact dwell timeout |
| four-paired-insert-04 | True | -0.00605 | completed probe |
| four-release-forecast-01 | False | -0.00825 | release_forecast: no verified bounded reference/withdrawal candidate |
| four-release-forecast-02 | False | -0.00825 | release_forecast: no verified bounded reference/withdrawal candidate |
| four-release-straight-01 | False | -0.00825 | release_forecast: no verified bounded reference/withdrawal candidate |
| four-release-taper-01 | False | -0.00825 | release_forecast: no verified bounded reference/withdrawal candidate |
| insert-probe-pressure-01 | False | 0.02064 | too many values to unpack (expected 4) |
| release-probe-held-01 | False | -1.63160 | completed probe |
| release-probe-held-wrist-01 | False | -1.63160 | completed probe |
| release-probe-legacy-01 | False | -7.85531 | completed probe |
| ten-centered-insert-01 | True | -0.00231 | completed probe |
| ten-first-insert-01 | False | 1.53436 | base_reposition: measured pose/stop dwell timeout |
| ten-paired-insert-01 | True | -0.00033 | completed probe |
| ten-thumb-opening-01 | False | -11.65803 | staging_clear: IK prefilter found no acceptable pose/path; predicted miss 0.000 m; replan required |
| ten-thumb-opening-02 | False | -11.65803 | finger_push: carton tilting; replan required |
| ten-thumb-opening-03 | False | 0.38206 | staging_clear: bounded staging complete; predicted remaining miss 0.006 m; replan required before grasping |
| ten-thumb-opening-04 | True | -0.02373 | completed probe |
| third-align-01 | False | -0.23125 | stage_align: supported position/yaw alignment timeout |
| third-align-02 | False | -0.04940 | release: withdrawal_unload_decision_budget_exhausted |
| third-insert-01 | False | 0.15290 | release: withdrawal_released_squared_support_or_clearance_not_verified |
| third-insert-02 | True | 0.01434 | completed probe |
| third-insert-03 | False | 0.12371 | release: withdrawal_bounded_nominal_geometry_or_motion |
| third-insert-04 | False | 0.11449 | release: withdrawal_unload_decision_budget_exhausted |
| third-insert-05 | True | 0.13815 | completed probe |
| third-withdraw-01 | False | unknown | unknown process |
| third-withdraw-02 | True | 0.14389 | completed probe |
| tight-three-01-capture | False | unknown | unknown process |
| tight-three-14-video | False | unknown | unknown process |
| tight-three-14-video-gpu | False | unknown | unknown process |

See [the dense task guide](../dense_stocking.md) for reproduction and limits.
