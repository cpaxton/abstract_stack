"""2D base navigation: a scene occupancy grid plus emet A* base path planning.

The grid is rebuilt from the live MuJoCo scene so randomized product positions
are reflected as obstacles. Planning is geometric only: it returns a collision
free (x, y) base route, never a tracking or contact guarantee.
"""

from __future__ import annotations

import numpy as np
from emet.motion.base.space import ConfigurationSpace


class SceneOccupancyGrid:
    """2D occupancy grid over the workspace, marked from the scene fixtures.

    Obstacles are the shelf, the supply table and every product's measured
    (possibly randomized) footprint. ``exp`` is the explored mask (here the whole
    allocated rectangle). Grid axes follow MuJoCo world X/Y; ``to_pt`` maps
    world (x, y) to integer (i, j) = (x cell, y cell).
    """

    def __init__(
        self,
        env,
        *,
        resolution: float = 0.02,
        bounds=((-1.2, -1.0), (0.6, 0.6)),
        margin: float = 0.03,
        exclude_products=(),
    ):
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Grid resolution must be finite and positive")
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Grid margin must be finite and nonnegative")
        self.resolution = float(resolution)
        self.margin = float(margin)
        (self.xmin, self.ymin), (self.xmax, self.ymax) = bounds
        if (
            not (self.xmin < self.xmax and self.ymin < self.ymax)
            or not np.isfinite([self.xmin, self.ymin, self.xmax, self.ymax]).all()
        ):
            raise ValueError("Grid bounds must be finite with xmin < xmax and ymin < ymax")
        self.nx = int(np.ceil((self.xmax - self.xmin) / self.resolution))
        self.ny = int(np.ceil((self.ymax - self.ymin) / self.resolution))
        self.exclude_products = set(exclude_products)
        if not self.exclude_products <= {p.name for p in env.products}:
            raise ValueError("Excluded products must be named scene products")
        self.obs: np.ndarray = np.zeros((self.nx, self.ny), dtype=bool)
        self.exp: np.ndarray = np.ones((self.nx, self.ny), dtype=bool)
        self.rebuild(env)

    def _mark_rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        x0, y0 = x0 - self.margin, y0 - self.margin
        x1, y1 = x1 + self.margin, y1 + self.margin
        if x1 < self.xmin or y1 < self.ymin or x0 > self.xmax or y0 > self.ymax:
            return
        i0 = max(0, int(np.floor((x0 - self.xmin) / self.resolution)))
        i1 = min(self.nx - 1, int(np.floor((x1 - self.xmin) / self.resolution)))
        j0 = max(0, int(np.floor((y0 - self.ymin) / self.resolution)))
        j1 = min(self.ny - 1, int(np.floor((y1 - self.ymin) / self.resolution)))
        if i0 <= i1 and j0 <= j1:
            self.obs[i0 : i1 + 1, j0 : j1 + 1] = True

    def rebuild(self, env) -> None:
        """Recompute the obstacle mask from the current scene state."""
        self.obs[:] = False
        config = env.config
        # Shelf bounding box (boards, sides and back).
        self._mark_rect(
            config.front_x - config.depth,
            config.center_y - config.width / 2,
            config.front_x,
            config.center_y + config.width / 2,
        )
        # Supply table.
        tx, ty = config.table_xy
        self._mark_rect(
            tx - config.table_size[0] / 2,
            ty - config.table_size[1] / 2,
            tx + config.table_size[0] / 2,
            ty + config.table_size[1] / 2,
        )
        # Measured product footprints (randomized positions).
        for product in env.products:
            if product.name in self.exclude_products:
                continue
            body = env.data.body(product.name)
            extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(product.half_size)
            self._mark_rect(
                body.xpos[0] - extent[0],
                body.xpos[1] - extent[1],
                body.xpos[0] + extent[0],
                body.xpos[1] + extent[1],
            )

    def to_pt(self, xy) -> tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        return (
            int(np.floor((x - self.xmin) / self.resolution)),
            int(np.floor((y - self.ymin) / self.resolution)),
        )

    def to_xy(self, pt) -> tuple[float, float]:
        i, j = int(pt[0]), int(pt[1])
        return (
            self.xmin + (i + 0.5) * self.resolution,
            self.ymin + (j + 0.5) * self.resolution,
        )

    def get_navigation_map(self):
        return self.obs, self.exp

    def get_2d_map(self):
        """Alias expected by ``emet`` planners that access ``space.voxel_map``."""
        return self.obs, self.exp

    @property
    def grid_resolution(self) -> float:
        return self.resolution


class BaseNavigationSpace(ConfigurationSpace):
    """A 2D (x, y) configuration space over a ``SceneOccupancyGrid``.

    Compatible with ``emet`` planners: it exposes ``get_navigation_map``,
    ``to_pt`` and ``to_xy``. ``is_valid`` uses a circular footprint radius for a
    cheap collision gate; the A* planner relies on the clearance field instead.
    """

    # MABEL's base footprint is roughly 0.38 m (x) by 0.56 m (y). A circular
    # clearance approximates the inscribed half-width; the actual oriented
    # footprint is enforced during execution by ``screen_base_translation``.
    FOOTPRINT_RADIUS_M = 0.22

    def __init__(self, grid: SceneOccupancyGrid, footprint_radius: float = FOOTPRINT_RADIUS_M):
        super().__init__(
            2,
            np.array([grid.xmin, grid.ymin]),
            np.array([grid.xmax, grid.ymax]),
            step_size=grid.resolution,
        )
        self.grid = grid
        # ``emet`` A* eagerly evaluates ``space.voxel_map.get_2d_map`` as the
        # default for ``get_navigation_map``; expose the grid under that name.
        self.voxel_map = grid
        self.footprint_radius = float(footprint_radius)
        if not np.isfinite(self.footprint_radius) or self.footprint_radius <= 0:
            raise ValueError("Footprint radius must be finite and positive")

    def to_pt(self, xy):
        return self.grid.to_pt(xy)

    def to_xy(self, pt):
        return self.grid.to_xy(pt)

    def get_navigation_map(self):
        return self.grid.get_navigation_map()

    def is_valid(self, state) -> bool:
        state = np.asarray(state, dtype=float)
        if state.shape != (3,) or not np.isfinite(state).all():
            return False
        pt = self.to_pt(state[:2])
        if not (0 <= pt[0] < self.grid.nx and 0 <= pt[1] < self.grid.ny):
            return False
        # Circular clearance gate: reject cells too close to any obstacle.
        radius_cells = int(np.ceil(self.footprint_radius / self.grid.resolution))
        i, j = pt
        lo_i, hi_i = max(0, i - radius_cells), min(self.grid.nx, i + radius_cells + 1)
        lo_j, hi_j = max(0, j - radius_cells), min(self.grid.ny, j + radius_cells + 1)
        window = self.grid.obs[lo_i:hi_i, lo_j:hi_j]
        if not window.any():
            return True
        yy, xx = np.nonzero(window)
        yy = yy + lo_j
        xx = xx + lo_i
        dist = np.hypot(xx - i, yy - j) * self.grid.resolution
        return bool(np.all(dist >= self.footprint_radius))


def coalesce_collinear_path(path):
    """Remove intermediate stops without changing the geometric polyline."""
    result = []
    for point in path:
        point = tuple(map(float, point))
        if result and np.linalg.norm(np.asarray(point) - result[-1]) < 1e-12:
            continue
        while len(result) >= 2:
            a = np.asarray(result[-1]) - result[-2]
            b = np.asarray(point) - result[-1]
            if (
                abs(a[0] * b[1] - a[1] * b[0]) > 1e-10 * np.linalg.norm(a) * np.linalg.norm(b)
                or a @ b < 0
            ):
                break
            result.pop()
        result.append(point)
    return result


def plan_base_path(
    env,
    start_xy,
    goal_xy,
    *,
    footprint_radius: float = BaseNavigationSpace.FOOTPRINT_RADIUS_M,
    resolution: float = 0.02,
    exclude_products=(),
    allow_goal_snap=False,
):
    """Plan a collision-free 2D base route with emet A*.

    Returns a list of (x, y) waypoints from ``start_xy`` to ``goal_xy``, or
    ``None`` when no route exists. Geometric planning only.
    """
    from emet.motion.algo.a_star import AStar

    grid = SceneOccupancyGrid(env, resolution=resolution, exclude_products=exclude_products)
    space = BaseNavigationSpace(grid, footprint_radius=footprint_radius)
    planner = AStar(
        space,
        min_clearance_m=footprint_radius,
        grid_resolution_m=grid.resolution,
        clearance_cost_weight=1.0,
    )
    result = planner.plan(
        [float(start_xy[0]), float(start_xy[1]), 0.0],
        [float(goal_xy[0]), float(goal_xy[1]), 0.0],
        verbose=False,
    )
    if not result.success or not result.trajectory:
        return None
    path = [tuple(float(v) for v in node.state[:2]) for node in result.trajectory]
    # Emet can explicitly resolve a low-clearance request to a nearby safe grid
    # endpoint. A successful planner result is not an exact-goal certificate.
    if not allow_goal_snap and not np.allclose(path[-1], goal_xy, rtol=0, atol=1e-9):
        return None
    return coalesce_collinear_path(path)
