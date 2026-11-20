# Small LfD collection pilot

Three fresh controller/sensing fixtures were collected in separate native processes. This is a pilot, not the full dataset, a trained policy or successful shelf stocking. All three task outcomes remain failure/no stocking attempted; training eligibility is false and fresh task credit is zero.

| Config seed | Reset jitter seed | Profile | Transitions | Raw RGB images | Wall time | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 1201 | 0 | lift_pan | 20/20 | 42/42 available | 12.35 s / 60 s | Process 0; task failure/no attempt |
| 1202 | 0 | left_wrist_pan | 20/20 | 42/42 available | 12.30 s / 60 s | Process 0; task failure/no attempt |
| 1203 | 0 | right_wrist_tilt | 20/20 | 42/42 available | 12.25 s / 60 s | Process 0; task failure/no attempt |

Totals: 60 transitions, 63 proprioception/F/T/tactile snapshots and 126 raw 160x120 RGB images. All camera sample times exactly match native pre-action boundaries. LeRobot contains 60 pre-action rows and 120 images; six terminal next-observation images remain in the neutral episodes.

The scene configuration seeds are 1201–1203. The frozen collector used the existing environment reset-jitter seed 0 for each. Distinct geometry is identified by each scene XML hash; the original records are retained unchanged.

| Profile | Measured excursion |
| --- | --- |
| lift_pan | Lift: 0.003917 m; Neck Yaw: 0.060930 rad |
| left_wrist_pan | Left Wrist: 0.029114 rad; Neck Yaw: 0.060938 rad |
| right_wrist_tilt | Right Wrist: 0.030389 rad; Neck Pitch: 0.053441 rad |

The requested trajectories are bounded half-sine actuator references and end at the initial nominal hold. Measured motion lags and is not fully settled at the final frame. These are controller fixtures, with no Cartesian IK/grasp admission or manipulation credit.

All-product native guards ran every 2 ms: 700 steps per process (500 settling + 200 recorded), 2100 total. No force guard or transient tip/drop incident occurred. Robot/fixture, robot/product and carton-pair normal-force peaks were 0 N. Native wrist parent-reaction measurements include unloaded gravity/inertial loads; contact-only wrench and tactile proxies were zero. All sensor channels were available.

## Export and failures

The first combined official LeRobot 0.7 export failed on its second episode: aggregate_stats interprets scalar feature names containing image as RGB statistics. Its partial output and traceback remain in run/lerobot and run/export.log. No native episode was dropped, recollected or relabeled.

The corrected adapter maps camera availability and sample times to scalar feature names outside that namespace and records the mapping in collection_provenance.json. Neutral episode names stay unchanged. Corrected export/reload passed for all 3 episodes / 60 rows, checking exact PNG pixels, reader float32/255 normalization, float64 native clock storage, nominal/applied separation, masks and outcomes. The updated verifier also reloads the retained older camera export without the new mapping.

No training or upload ran. No video codec was tested. Complete manipulation collection remains a later integration task.

## Reproduction and source

Base revision: 03bea50192853908e225bbbd666323360cafbb75; source is an explicitly dirty snapshot. Full source copies and hashes are in run/source_snapshot and run/source_sha256.json. The controller and model were not changed. The shared /tmp/abstract-stack-render.lock was held for native/render-heavy checks and released before CPU export.

Commands, independent process outcomes and wall budgets are retained in run/process.json and export-corrected/process.json. The supervisor is collect_pilot.py. Each collection used --steps 20 --cameras --seed CONFIG_SEED --motion-profile PROFILE with a 60-second wall budget; SDK stages had 45-second budgets. See report.json for per-channel motion, source/artifact hashes, guard results and all limitations.

The pilot collector/exporter follow-up was developed separately in `/tmp/abstract-stack-lfd-pr` on top of `03bea50`; the frozen code under `outputs/lfd-pilot-v1/` identifies exactly what ran. These artifacts do not imply that its follow-up patch has been merged into this checkpoint branch. Neutral episode schemas are unchanged. The native collector/adapter worktree passed 28 focused tests and changed-file hooks.

The follow-up code is now saved locally as `f2fded7` on `feat/lfd-pilot` (not published). The historical frozen collection source and its dirty-snapshot hashes remain unchanged.
