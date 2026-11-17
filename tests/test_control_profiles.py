"""Force profiles preserve defaults and remain independent of policy switches."""

import json
import subprocess
import sys

import pytest

from abstract_stack.campaign import run_campaign
from abstract_stack.cli import _parser
from abstract_stack.control_profiles import force_profile_settings, resolve_control_settings
from abstract_stack.interactions import ScriptConfig


def test_force_profiles_and_explicit_override_precedence(tmp_path):
    defaults = ScriptConfig()
    assert force_profile_settings() == {
        "grasp_force_target": defaults.grasp_force_target,
        "push_force_target": defaults.push_force_target,
    }
    assert force_profile_settings("legacy") == {
        "grasp_force_target": 6.0,
        "push_force_target": 2.0,
    }
    config = tmp_path / "overrides.json"
    config.write_text(json.dumps({"push_force_target": 1.0}))
    assert resolve_control_settings("legacy", config) == {
        "grasp_force_target": 6.0,
        "push_force_target": 1.0,
    }
    config.write_text(json.dumps({"max_force": 1000}))
    with pytest.raises(ValueError, match="Not a tunable"):
        resolve_control_settings("legacy", config)
    with pytest.raises(ValueError, match="Unknown force profile"):
        force_profile_settings("typo")


@pytest.mark.parametrize("command", ["script", "solve", "batch"])
def test_profile_does_not_change_contact_or_hand_modes(command):
    args = _parser().parse_args([command, "--force-profile", "legacy"])
    assert args.force_profile == "legacy"
    assert not args.legacy_contact_control
    assert args.thumb_preshape
    assert _parser().parse_args([command]).force_profile == "selected"


def test_campaign_records_and_forwards_profile_even_on_timeout(tmp_path, monkeypatch):
    captured = []
    real_run = subprocess.run

    def timeout(command, **kwargs):
        if command[0] != sys.executable:
            return real_run(command, **kwargs)
        captured.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("abstract_stack.campaign.subprocess.run", timeout)
    campaign = run_campaign(tmp_path, [42], force_profile="legacy", episode_timeout=1)
    assert campaign["force_profile"] == "legacy"
    assert campaign["force_profile_settings"] == force_profile_settings("legacy")
    assert campaign["control_overrides"] == {}
    command = captured[0]
    assert command[command.index("--force-profile") + 1] == "legacy"
    assert "--legacy-contact-control" not in command
    assert campaign["runs"][0]["returncode"] == 124
    assert campaign["runs"][0]["physical_success"] is None
