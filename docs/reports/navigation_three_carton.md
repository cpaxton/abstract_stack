# Base navigation and the first three-carton episode

2026-11-18. Perfect-state development on one fixed shelf. The grasp-skill
controller now relocates its base through native wheels when a loaded shelf
route is out of reach, planned on a scene occupancy grid with `emet` A*, and
completes **three consecutive stocks** in a single episode. No student has been
trained and no dense ten-carton or general robustness claim is made.

## Result

| Fixed three-carton episode | Result |
| --- | --- |
| Pick / shelf support / final position / retreat | 3 / 3 at every stage |
| Prefills / tip-drop incidents | 0 / 0 |
| Peak robot-active-product normal | 8.06 N |
| Peak robot-fixture normal | 8.78 N |
| Base relocations | 1 (loaded, 0.029 m, 2 waypoints) |

Reproduction:

```bash
MABEL_ROOT=third_party/open_mabel flock /tmp/abstract-stack-render.lock \
  .venv/bin/python -m abstract_stack.cli script \
  --config outputs/grasp-skill-v1/three-carton-06/scene.json --seed 0 \
  --grasp-skill --base-reposition \
  --output-dir outputs/three-carton-astar-02/episode
```

The episode is `outputs/three-carton-astar-02/episode`. It is the first run to
schedule and complete all three cartons; earlier two-carton and blocked
three-carton results remain in their own outputs.

## Why the third carton failed before

`carton_2` is placed into the near-centreline slot (`y = -0.025`). The left
arm's `Left Arm 2` adduction range bottoms out at `-0.175 rad`, so the loaded
shelf approach is unreachable with the base parked at its grasp pose. Two more
findings follow from that single limit:

- The idle right hand rests ~8 mm from the shelf's right side panel at home, so
  *any* base translation swept it into the panel. It is now swung outward
  (`Right Arm 3 ≈ +0.5`) before the base moves.
- `carton_0` (the far supply carton) is reachable at the grasp endpoint but its
  joint-interpolated pregrasp path grazed the shelf's left side panel. The
  pregrasp move now uses the same checked Cartesian entry as the grasp descent.

## Base navigation

`abstract_stack.navigation` builds a 2D occupancy grid from the live scene
(shelf, supply table and every product's measured footprint, so randomized
product positions become obstacles) and plans a collision-free base route with
`emet`'s A* planner (`SceneOccupancyGrid` + `BaseNavigationSpace` +
`plan_base_path`). The carried product is excluded from the static map and
tracked relative to the hand during the move. Execution follows the planned
waypoints through the native swerve controller under the existing
`BaseContactGuard` force/tip/yaw guards and the carried-object slip check.

The planner uses a circular clearance approximation (0.22 m) for MABEL's
roughly 0.38 m x 0.56 m base; the executed waypoints are still protected by
measured force and grasp-slip guards rather than the geometric screen alone.
`emet.mapping` (the upstream voxel/occupancy modules) is not importable in this
environment because its `rerun`/`pyarrow` chain requires NumPy 2 while the
project pins NumPy 1.26.4; only `emet.motion` is used here. See
[home_robot_issues.md](../home_robot_issues.md) (HR-005).

## Scope

The base relocation is admitted only by the opt-in `--base-reposition` flag and
only for the grasp-skill controller on the fixed shelf. It is a kinematic
necessary condition plus measured-force execution guards, not a proof of
arbitrary-order or arbitrary-pose success. The three cartons are completed with
zero prefills; the ten-carton dense goal and the rear-clearance task remain
open.
