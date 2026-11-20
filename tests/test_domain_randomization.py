"""DR determinism, valid geometry, split isolation and all-object evidence."""

import hashlib
import json
import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from abstract_stack.config import load_scene_config
from abstract_stack.domain_randomization import FAMILIES, generate_suite, sample_scene
from abstract_stack.scene import initial_positions, products_for


@pytest.mark.parametrize("family", FAMILIES)
def test_repeated_samples_preserve_geometry_counts_and_robot_contract(family):
    for seed in range(20):
        config, context = sample_scene(seed, family)
        again, other = sample_scene(seed, family)
        assert asdict(config) == asdict(again) and context == other
        products = products_for(config)
        assert 2 <= len(products) <= 10 and len({p.name for p in products}) == len(products)
        assert context["scheduled_product_names"] == [p.name for p in products]
        assert not any(p.initially_stocked for p in products)
        assert 1 <= len(config.levels) <= 3
        assert (
            context["factors"]["whole_row_free_space_m"] in (0.005, 0.01)
            if family == "dense_cartons"
            else config.row_free_space is None
        )
        assert config.target_tolerance == 0.035
        if family == "mixed_house":
            assert {p.kind for p in products} == {
                "carton",
                "bottle",
            } and config.environment == "small_house"


def test_suite_reproducibility_hashes_and_disjoint_splits(tmp_path):
    a = generate_suite(
        tmp_path / "a", development_count=4, heldout_count=4, development_seed=31, heldout_seed=32
    )
    b = generate_suite(
        tmp_path / "b", development_count=4, heldout_count=4, development_seed=31, heldout_seed=32
    )
    assert {p.relative_to(a): p.read_bytes() for p in a.rglob("*.json")} == {
        p.relative_to(b): p.read_bytes() for p in b.rglob("*.json")
    }
    seeds = []
    for split in ("development", "heldout"):
        index = json.loads((a / split / "index.json").read_text())
        seen = set()
        for case in index["cases"]:
            context = json.loads((a / case["context"]).read_text())
            assert context["split"] == split
            assert (
                hashlib.sha256((a / case["scene"]).read_bytes()).hexdigest()
                == context["scene_config_sha256"]
            )
            seen.add(context["scene_seed"])
        seeds.append(seen)
        ledger = json.loads((a / split / "evaluation.json").read_text())
        assert all(
            c["outcome"] == "not_run" and not any(any(p.values()) for p in c["products"].values())
            for c in ledger["cases"]
        )
    assert seeds[0].isdisjoint(seeds[1])


def test_pose_suite_varies_reset_positions_and_keeps_fixed_row(tmp_path):
    base = load_scene_config(
        Path(__file__).resolve().parents[1] / "configs/scenes/tight_ten_cartons_20mm.json"
    )
    suite = generate_suite(
        tmp_path / "poses",
        development_count=8,
        heldout_count=1,
        development_seed=29,
        pose_base=base,
    )
    index = json.loads((suite / "development/index.json").read_text())
    shelf_positions, supply_positions = set(), []
    for row in index["cases"]:
        config = load_scene_config(suite / row["scene"])
        context = json.loads((suite / row["context"]).read_text())
        nx, ny = context["factors"]["shelf_translation_from_base_m"]
        assert -0.012 <= nx <= 0.004 and -0.015 <= ny <= 0.015
        assert config.table_xy == base.table_xy
        assert config.supply_xy == base.supply_xy
        assert config.row_free_space == 0.020 and config.initially_stocked == 0
        assert config.product_count == 10 and config.carton_size == base.carton_size
        assert config.levels == base.levels and config.robot_xy == base.robot_xy
        assert 0.002 <= config.spawn_jitter <= 0.006
        products = products_for(config)
        assert len(products) == 10
        reset = initial_positions(config, products, context["reset_seed"])
        np.testing.assert_array_equal(
            reset, initial_positions(config, products, context["reset_seed"])
        )
        shelf_positions.add((config.front_x, config.center_y))
        supply_positions.append(reset)
    assert len(shelf_positions) == 8
    assert not np.array_equal(supply_positions[0], supply_positions[1])


def test_default_evaluation_entropy_is_independent_of_development(tmp_path, monkeypatch):
    entropy = iter((1001, 1002))
    monkeypatch.setattr(
        "abstract_stack.domain_randomization.secrets.randbits", lambda n: next(entropy)
    )
    a = generate_suite(tmp_path / "a", development_count=1, heldout_count=1)
    b = generate_suite(tmp_path / "b", development_count=1, heldout_count=1)
    assert (a / "development/index.json").read_bytes() == (
        b / "development/index.json"
    ).read_bytes()
    assert (a / "heldout/index.json").read_bytes() != (b / "heldout/index.json").read_bytes()


@pytest.mark.parametrize(
    "seed,family", [(-1, "bench_cartons"), (True, "bench_cartons"), (1, "unknown")]
)
def test_invalid_sampling_requests_rejected(seed, family):
    with pytest.raises(ValueError):
        sample_scene(seed, family)


def test_grade_cannot_drop_objects_relabel_seeds_or_trust_truthy_stage_strings():
    grade = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/evaluate_domain_suite.py")
    )["grade_result"]
    stages = dict.fromkeys(("pick", "place", "final_position", "retreat"), True)
    result = dict(
        scheduled_pick_products=["carton_0", "carton_1"],
        metrics=dict(seed=4),
        task_success=True,
        physical_success=True,
        stage_evaluation=dict(
            no_knockovers=True, products={"carton_0": stages.copy(), "carton_1": stages.copy()}
        ),
    )
    assert grade(result, ["carton_0", "carton_1"], 4)["clean_success"]
    assert not grade(result, ["carton_0", "carton_1"], 5)["clean_success"]
    result["stage_evaluation"]["products"]["carton_1"]["retreat"] = "true"
    assert not grade(result, ["carton_0", "carton_1"], 4)["clean_success"]
    result["scheduled_pick_products"] = ["carton_0"]
    assert (
        grade(result, ["carton_0", "carton_1"], 4)["outcome"] == "invalid_scheduled_object_evidence"
    )
