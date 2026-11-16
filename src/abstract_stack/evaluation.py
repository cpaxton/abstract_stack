"""Read-only stage evidence, including transient disturbances of every product."""

from __future__ import annotations

import numpy as np


class StageObserver:
    def __init__(self, env):
        self.env = env
        self.incidents = []
        self.seen = set()
        self.completed = set()
        self.retreat = None
        self.retreat_results = {}

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
            final = next((e for e in own if e.get("verified_release")), {})
            retreat = self.retreat_results.get(p.name, {})
            products[p.name] = {
                "attempted": any(e["phase"] == "grasp_precheck" for e in own),
                "pick": any(e.get("verified_lift", False) for e in own),
                "place": any(e.get("verified_shelf_support", False) for e in own),
                "final_position": bool(final.get("placement_conditions", False))
                and any(e.get("back_contact_verified", False) for e in own),
                "retreat": bool(retreat.get("verified", False)),
                "retreat_evidence": retreat or None,
            }
        return {
            "products": products,
            "incidents": self.incidents,
            "no_knockovers": not self.incidents,
            "unfinished_retreat": self.retreat,
            "final_state": self.snapshot(),
        }
