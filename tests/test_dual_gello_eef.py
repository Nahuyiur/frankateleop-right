import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dual_gello_eef", ROOT / "scripts/dual_gello_eef.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config(tmp_path, verified=False):
    value = json.loads((ROOT / "config/gello_eef_dual.json").read_text())
    for side in ("left", "right"):
        value[side]["calibration_verified"] = verified
        if verified:
            value[side]["workspace_base_m"] = {"x": [.2, .7], "y": [-.3, .3], "z": [-.4, .4]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return path


def test_shipped_config_is_preview_only(tmp_path):
    path = config(tmp_path)
    MODULE.load_config(path)
    with pytest.raises(ValueError, match="verify base mapping"):
        MODULE.load_config(path, execute=True)


def test_distinct_serial_endpoints_and_remote_execution(tmp_path):
    value = MODULE.load_config(config(tmp_path, True), True)
    left = MODULE.arm_command(value, "left", True)
    right = MODULE.arm_command(value, "right", True)
    assert "--supervised" in left and "--execute" in left
    assert "LEFT_TELEOP_NODE_PORT=6002" in left
    assert any("FTBZIB2R" in x for x in left)
    assert right[0] == "ssh" and "pnp@192.168.100.86" in right
    assert "B_run_right_gello_eef.sh" in right[-1] and "FTBJTKP2" in right[-1]
    assert "RIGHT_TELEOP_NODE_PORT=6001" in right[-1]


@pytest.mark.parametrize("side", ["left", "right"])
def test_one_unverified_arm_blocks_all_execute(tmp_path, side):
    path = config(tmp_path, True)
    value = json.loads(path.read_text())
    value[side]["calibration_verified"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        MODULE.load_config(path, True)


FAKE = r'''
import pathlib, sys, time
side, output, delay, fail = sys.argv[1:]
time.sleep(float(delay))
print('GELLO_EEF_READY ' + side, flush=True)
path = pathlib.Path(output)
for line in sys.stdin:
    if line.strip() == 'GO':
        with path.open('a') as f: f.write('GO ' + str(time.monotonic()) + '\n')
        if fail == 'yes':
            time.sleep(.1)
            sys.exit(7)
    if line.strip() == 'STOP':
        with path.open('a') as f: f.write('STOP\n')
        break
'''


def fake_commands(tmp_path, fail="no", right_delay=.15):
    return {side: [sys.executable, "-u", "-c", FAKE, side, str(tmp_path / side),
                   str(right_delay if side == "right" else 0), fail if side == "left" else "no"]
            for side in ("left", "right")}


def test_two_ready_barrier_and_clean_stop_with_fake_processes(tmp_path):
    started = time.monotonic()
    assert MODULE.supervise(fake_commands(tmp_path), startup_timeout=2, duration=.7) == 0
    for side in ("left", "right"):
        lines = (tmp_path / side).read_text().splitlines()
        assert float(lines[0].split()[1]) - started >= .15
        assert lines[1] == "STOP"


def test_failure_of_one_fake_arm_stops_other(tmp_path):
    with pytest.raises(RuntimeError, match="stopping both"):
        MODULE.supervise(fake_commands(tmp_path, fail="yes"), startup_timeout=2)
    assert "STOP" in (tmp_path / "right").read_text()


def test_no_go_when_one_arm_not_ready(tmp_path):
    with pytest.raises(TimeoutError):
        MODULE.supervise(fake_commands(tmp_path, right_delay=.4), startup_timeout=.1)
    for side in ("left", "right"):
        path = tmp_path / side
        assert not path.exists() or "GO" not in path.read_text()


def test_right_node_wrapper_does_not_home_or_open_gripper():
    text = (ROOT / "right_franka/3_launch_eef_node.sh").read_text()
    for flag in ("--control-mode ee", "--no-home-on-init", "--no-open-gripper-on-init", "--no-move-to-initial-pose"):
        assert flag in text
    assert "kill_port" not in text
