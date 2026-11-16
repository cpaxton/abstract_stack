# Dense insertion development trials

These are successive development trials with different policies or initial conditions. They are not a frozen-policy multi-scene success-rate study. Full videos and traces remain under ignored `outputs/`; the companion JSON pins sources and artifacts.

Each row has ten cartons. Prefilled cartons earn no pick/place/retreat credit. All products are monitored at each physics step for transient tips and drops. A successful row requires all scheduled stages and the whole-shelf packing dwell.

| Trial | Total free mm | Prefilled | Clean | Pick/place/final/retreat | Outcome |
|---|---:|---:|---|---|---|
| dense-ten-10mm-insertion-v6 | 10 | 9 | no | 1/1/0/0 / 1 scheduled | finger_push: contact dwell timeout |
| dense-ten-10mm-insertion-v7 | 10 | 9 | no | 1/1/0/0 / 1 scheduled | finger_push: contact dwell timeout |
| dense-ten-10mm-insertion-v8 | 10 | 9 | no | 1/1/0/0 / 1 scheduled | finger_push: feedback path rejected before actuation |
| dense-ten-10mm-insertion-v9 | 10 | 9 | no | 1/1/0/0 / 1 scheduled | finger_push: feedback path rejected before actuation |
| dense-ten-10mm-insertion-v10 | 10 | 9 | yes | 1/1/1/1 / 1 scheduled | complete |
| dense-ten-5mm-insertion-v1 | 5 | 9 | no | 1/1/0/0 / 1 scheduled | Left Index MCP: -0.34907000000000005 is outside [-0.34907, 1.65806] |
| dense-ten-5mm-insertion-v2 | 5 | 9 | yes | 1/1/1/1 / 1 scheduled | complete |
| dense-ten-10mm-two-insertions-v1 | 10 | 8 | no | 1/1/0/0 / 2 scheduled | finger_push: feedback path rejected before actuation |
| dense-ten-10mm-two-insertions-v2 | 10 | 8 | yes | 2/2/2/2 / 2 scheduled | complete |
| dense-ten-10mm-video-v1 | 10 | 9 | yes | 1/1/1/1 / 1 scheduled | complete |
| dense-ten-10mm-full-v1 | 10 | 0 | no | 1/0/0/0 / 10 scheduled | align_shelf: IK prefilter found no acceptable pose/path; predicted miss 0.094 m; replan required |
| seed-1 | 5 | 9 | no | 1/0/0/0 / 1 scheduled | transfer: IK prefilter found no acceptable pose/path; predicted miss 0.010 m; replan required |
| seed-2 | 5 | 9 | no | 0/0/0/0 / 1 scheduled | pregrasp: contact force exceeded 25.0 N |
| seed-3 | 5 | 9 | no | 1/1/0/0 / 1 scheduled | finger_push: contact dwell timeout |

See [the dense task guide](../dense_stocking.md) for reproduction and limits.
