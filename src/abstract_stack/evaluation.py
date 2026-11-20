"""Read-only stage evidence, including transient disturbances of every product."""

from __future__ import annotations

import numpy as np


def strict_final_position(events):
    """Use the latest verified release and the measured back-contact witness."""
    release = next((event for event in reversed(events) if event.get("verified_release")), {})
    terminal = next(
        (event for event in reversed(events) if event["phase"] == "dense_final_pose_recheck"), {}
    )
    return (
        bool(release.get("placement_conditions", False))
        and release.get("strict_release_yaw_verified", True)
        and any(event.get("back_contact_verified", False) for event in events)
        and terminal.get("verified", True)
    )


class StageObserver:
    def __init__(self, env):
        self.env = env
        self.incidents = []
        self.seen = set()
        self.completed = set()
        self.retreat = None
        self.retreat_results = {}
        self.initial = self.snapshot()
        self.maximum_displacements = {p.name: 0.0 for p in env.products}

    def snapshot(self):
        good = self.env._placement_conditions()
        return {
            p.name: {
                "pose": np.r_[
                    self.env.data.body(p.name).xpos, self.env.data.body(p.name).xquat
                ].tolist(),
                "placement_conditions": bool(good[i]),
                "stocked": bool(self.env._stocked[i]),
            }
            for i, p in enumerate(self.env.products)
        }

    def sample(self, phase, active):
        env = self.env
        for p in env.products:
            body = env.data.body(p.name)
            self.maximum_displacements[p.name] = max(
                self.maximum_displacements[p.name],
                float(np.linalg.norm(body.xpos - np.asarray(self.initial[p.name]["pose"][:3]))),
            )
            conditions = {
                "tipped": body.xmat.reshape(3, 3)[2, 2] < np.cos(np.radians(45)),
                "dropped": body.xpos[2] < 0.25,
            }
            for kind, occurred in conditions.items():
                key = (p.name, kind, phase, active)
                if occurred and key not in self.seen:
                    self.seen.add(key)
                    self.incidents.append(
                        {
                            "product": p.name,
                            "kind": kind,
                            "phase": phase,
                            "active_product": active,
                            "sim_time": float(env.data.time),
                        }
                    )
            if self.retreat is not None:
                origin = np.asarray(self.retreat["before"][p.name]["pose"][:3])
                displacement = float(np.linalg.norm(body.xpos - origin))
                self.retreat["max_displacement_m"][p.name] = max(
                    self.retreat["max_displacement_m"][p.name], displacement
                )

    def begin_retreat(self, product):
        self.retreat = {
            "product": product,
            "before": self.snapshot(),
            "incident_start": len(self.incidents),
            "max_displacement_m": {p.name: 0.0 for p in self.env.products},
        }

    def end_retreat(self, product):
        evidence = self.retreat
        after = self.snapshot()
        incidents = self.incidents[evidence.pop("incident_start") :]
        required = self.completed | {product}
        evidence.update(
            after=after,
            incidents=incidents,
            verified=not incidents and all(after[n]["stocked"] for n in required),
        )
        self.retreat_results[product] = evidence
        if evidence["verified"]:
            self.completed.add(product)
        self.retreat = None

    def report(self, events):
        products = {}
        for p in self.env.products:
            own = [e for e in events if e["product"] == p.name]
            retreat = self.retreat_results.get(p.name, {})
            products[p.name] = {
                "initially_stocked": p.initially_stocked,
                "attempted": any(e["phase"] == "grasp_precheck" for e in own),
                "pick": any(e.get("verified_lift", False) for e in own),
                "place": any(e.get("verified_shelf_support", False) for e in own),
                "final_position": strict_final_position(own),
                "retreat": bool(retreat.get("verified", False)),
                "retreat_evidence": retreat or None,
            }
        return {
            "products": products,
            "incidents": self.incidents,
            "no_knockovers": not self.incidents,
            "unfinished_retreat": self.retreat,
            "final_state": self.snapshot(),
            "initial_state": self.initial,
            "maximum_displacement_from_start_m": self.maximum_displacements,
        }


def manipulation_evaluation(result):
    """Grade scheduled picks separately from explicit prefilled initial conditions."""
    names = result["scheduled_pick_products"]
    products = result["stage_evaluation"]["products"]
    counts = {
        stage: sum(bool(products[name][stage]) for name in names)
        for stage in ("pick", "place", "final_position", "retreat")
    }
    packed = (
        bool(names)
        and result["physical_success"]
        and result["stage_evaluation"]["no_knockovers"]
        and all(counts[stage] == len(names) for stage in ("pick", "place", "retreat"))
    )
    return {
        "scheduled_manipulations": len(names),
        "scheduled_product_names": names,
        "prefilled_product_names": result["initially_stocked_products"],
        "stages": counts,
        "packed_success": packed,
        "clean_success": packed and counts["final_position"] == len(names),
        "remaining_alignment_products": [
            name for name in names if not products[name]["final_position"]
        ],
    }


def reported_manipulations(run):
    """Recompute stage credit, including historical summaries without pick lists.

    Process/artifact validity remains the caller's responsibility. Never trust
    a cached success flag instead of the individual manipulation evidence.
    """
    evaluation = run.get("stage_evaluation") or {}
    products = evaluation.get("products", {})
    names = run.get("scheduled_pick_products")
    if names is None:
        names = [name for name, p in products.items() if not p.get("initially_stocked", False)]
    counts = {
        stage: sum(bool(products.get(name, {}).get(stage)) for name in names)
        for stage in ("pick", "place", "final_position", "retreat")
    }
    return {
        "names": names,
        "stages": counts,
        "clean_success": bool(names)
        and len(set(names)) == len(names)
        and set(names).issubset(products)
        and len(products) == run.get("product_count", len(products))
        and run.get("physical_success") is True
        and run.get("task_success", True) is True
        and evaluation.get("no_knockovers") is True
        and all(count == len(names) for count in counts.values()),
    }
