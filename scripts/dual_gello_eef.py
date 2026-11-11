#!/usr/bin/env python3
"""Supervise two existing local GELLO EEF stacks; no robot SDK in this process."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
SHARED_FILES = (
    "A_run_left_gello_eef.sh", "B_run_right_gello_eef.sh",
    "left_franka/4_run_eef_gello.sh", "right_franka/3_launch_eef_node.sh",
    "right_franka/4_run_eef_gello.sh", "teleop/experiments/gello_eef_teleop.py",
)


def vector(value, size, label):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(label + " must be a list of length " + str(size))
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(label + " must contain finite numbers")
    return value


def load_config(path, execute=False):
    config = json.loads(Path(path).read_text())
    for key in ("translation_scale", "rotation_scale", "max_anchor_translation", "max_anchor_rotation"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
    for key in ("right_repo", "ssh_identity"):
        if not isinstance(config.get(key), str) or not config[key].startswith("/"):
            raise ValueError(key + " must be an absolute path")
    target = config.get("right_ssh", "")
    if not isinstance(target, str) or target.startswith("-") or "@" not in target or any(c.isspace() for c in target):
        raise ValueError("right_ssh must be user@host")
    for side in ("left", "right"):
        arm = config[side]
        vector(arm.get("leader_to_follower_base_rpy_deg"), 3, side + " base alignment")
        if not isinstance(arm.get("calibration_verified"), bool):
            raise ValueError(side + " calibration_verified must be boolean")
        if not isinstance(arm.get("teleop_port"), str) or not arm["teleop_port"].startswith("/dev/"):
            raise ValueError(side + " requires an explicit calibrated serial device")
        workspace = arm.get("workspace_base_m")
        if workspace is not None:
            for axis in ("x", "y", "z"):
                low, high = vector(workspace.get(axis), 2, side + " workspace " + axis)
                if low >= high:
                    raise ValueError(side + " workspace requires LOW < HIGH")
        if execute and (arm["calibration_verified"] is not True or workspace is None):
            raise ValueError(side + ": verify base mapping, leader FK/TCP and workspace_base_m before --execute")
    return config


def ssh_prefix(config):
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "ServerAliveInterval=1", "-o", "ServerAliveCountMax=3",
            "-i", config["ssh_identity"], config["right_ssh"]]


def arm_command(config, side, execute):
    repo = str(ROOT) if side == "left" else config["right_repo"]
    arm = config[side]
    script = "A_run_left_gello_eef.sh" if side == "left" else "B_run_right_gello_eef.sh"
    prefix = side.upper()
    ports = (50052, 50054, 6002) if side == "left" else (50051, 50053, 6001)
    command = ["env", "PYTHONUNBUFFERED=1", "GELLO_EEF_SIDE=" + side,
               prefix + "_TELEOP_PORT=" + arm["teleop_port"],
               prefix + "_ROBOT_SERVER_PORT=" + str(ports[0]),
               prefix + "_GRIPPER_SERVER_PORT=" + str(ports[1]),
               prefix + "_TELEOP_NODE_PORT=" + str(ports[2]),
               "bash", repo + "/" + script, "--supervised",
               "--base-alignment-rpy-deg"]
    command += [str(x) for x in arm["leader_to_follower_base_rpy_deg"]]
    for key in ("translation_scale", "rotation_scale", "max_anchor_translation", "max_anchor_rotation"):
        command += ["--" + key.replace("_", "-"), str(config[key])]
    if execute:
        command += ["--execute", "--calibration-verified"]
    workspace = arm.get("workspace_base_m")
    if workspace is None:
        command += ["--preview-anchor-only"]
    else:
        for axis in ("x", "y", "z"):
            command += ["--workspace-" + axis] + [str(x) for x in workspace[axis]]
    return command if side == "left" else ssh_prefix(config) + [shlex.join(command)]


def preflight(config):
    """Read-only host checks. Never sync code, kill ports or open serial devices."""
    probe = """
import hashlib, json, socket, sys
from pathlib import Path
data = json.loads(sys.argv[1])
root = Path(data['repo'])
for name, digest in data['files'].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit('EEF code missing/mismatched: ' + str(path))
for name in data['required']:
    if not (root / name).is_file():
        raise SystemExit('Required stack file missing: ' + name)
if not Path(data['serial']).exists():
    raise SystemExit('GELLO serial missing: ' + data['serial'])
for port in data['ports']:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise SystemExit('Port already used; stop existing stack yourself: ' + str(port))
print('EEF host preflight OK (no device opened)')
"""
    digests = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SHARED_FILES}
    for side in ("left", "right"):
        data = {
            "repo": str(ROOT) if side == "left" else config["right_repo"],
            "files": digests,
            "required": [side + "_franka/" + f for f in ("1_launch_robot.sh", "2_launch_gripper.sh", "3_launch_node.sh")],
            "serial": config[side]["teleop_port"],
            "ports": [50052, 50054, 6002] if side == "left" else [50051, 50053, 6001],
        }
        command = ["/usr/bin/python3", "-B", "-c", probe, json.dumps(data)]
        if side == "right":
            command = ssh_prefix(config) + [shlex.join(command)]
        subprocess.run(command, check=True, timeout=12, stdin=subprocess.DEVNULL)


def supervise(commands, startup_timeout=400.0, duration=0.0):
    """Two-ready barrier plus lease; either child exit shuts down both stacks."""
    children = {}
    events = queue.Queue()
    stop = threading.Event()
    ready = set()
    go = False
    old_handlers = {}

    def read_output(side, pipe):
        for line in pipe:
            events.put((side, line.rstrip()))

    def send(proc, message):
        proc.stdin.write(message + "\n")
        proc.stdin.flush()

    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
        for side, command in commands.items():
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    start_new_session=True)
            children[side] = proc
            os.set_blocking(proc.stdin.fileno(), False)
            threading.Thread(target=read_output, args=(side, proc.stdout), daemon=True).start()
        started = time.monotonic()
        last_lease = 0.0
        while not stop.is_set():
            now = time.monotonic()
            if duration > 0 and now - started >= duration:
                return 0
            for side, proc in children.items():
                if proc.poll() is not None:
                    raise RuntimeError(side + " stack exited (" + str(proc.returncode) + "); stopping both")
            # Bound output processing so a noisy child cannot starve the lease.
            for _ in range(100):
                try:
                    side, line = events.get_nowait()
                except queue.Empty:
                    break
                print("[" + side + "] " + line, flush=True)
                if line == "GELLO_EEF_READY " + side:
                    ready.add(side)
            if not go and ready == set(commands):
                for proc in children.values():
                    send(proc, "GO")
                go = True
                print("Both arms ready; GO sent. Each arm captures its own live anchor.", flush=True)
            if not go and now - started > startup_timeout:
                raise TimeoutError("Both arms did not become ready before startup timeout")
            if now - last_lease >= 0.2:
                for proc in children.values():
                    send(proc, "LEASE")
                last_lease = now
            stop.wait(0.05)
        return 130
    finally:
        # STOP/EOF reaches the remote adapter over SSH even when it is outside
        # the local process group. Lease expiry handles a broken SSH transport.
        for proc in children.values():
            try:
                send(proc, "STOP")
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        deadline = time.monotonic() + 8.0
        for proc in children.values():
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in children.values():
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="local unit tests only; no SSH or hardware")
    mode.add_argument("--plan", action="store_true", help="show commands; no SSH or hardware")
    mode.add_argument("--preview", action="store_true", help="start both hardware stacks, but send no EEF/gripper targets")
    mode.add_argument("--execute", action="store_true", help="requires verified per-arm mapping and workspace")
    parser.add_argument("--config", type=Path, default=ROOT / "config/gello_eef_dual.json")
    parser.add_argument("--startup-timeout", type=float, default=400)
    parser.add_argument("--duration-sec", type=float, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("startup-timeout must be finite and positive")
    if not math.isfinite(args.duration_sec) or args.duration_sec < 0:
        parser.error("duration-sec must be finite and non-negative")
    if not any((args.check, args.plan, args.preview, args.execute)):
        parser.print_help()
        return 0
    config = load_config(args.config, args.execute)
    if args.check:
        return subprocess.call(["bash", str(ROOT / "A_run_left_gello_eef.sh"), "--check"])
    commands = {side: arm_command(config, side, args.execute) for side in ("left", "right")}
    if args.plan:
        for side, command in commands.items():
            print(side + ": " + shlex.join(command))
        return 0
    preflight(config)
    print("Mode: " + ("EXECUTE" if args.execute else "PREVIEW (hardware services WILL start)"), flush=True)
    return supervise(commands, args.startup_timeout, args.duration_sec)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, TimeoutError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit("GELLO EEF STOP: " + str(exc))
