"""DR determinism, valid geometry, split isolation and all-object evidence."""

import hashlib
import json
import runpy
from dataclasses import asdict
from pathlib import Path

import pytest

from abstract_stack.domain_randomization import FAMILIES, generate_suite, sample_scene
from abstract_stack.scene import products_for


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
