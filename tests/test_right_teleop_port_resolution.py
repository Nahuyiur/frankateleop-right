import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVER = REPO_ROOT / "scripts" / "resolve_right_teleop_port.sh"


def run_resolver(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("RIGHT_TELEOP_PORT", None)
    env.pop("FRANKA_TELEOP_PORT", None)
    env.pop("TELEOP_PORT", None)
    env.update(overrides)
    env["SERIAL_BY_ID_DIR"] = str(tmp_path)
    return subprocess.run(
        ["bash", str(RESOLVER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def make_port(path: Path) -> Path:
    path.touch(mode=0o600)
    return path


def test_prefers_configured_primary(tmp_path: Path) -> None:
    primary = make_port(tmp_path / "primary")

    result = run_resolver(tmp_path, RIGHT_TELEOP_PRIMARY_PORT=str(primary))

    assert result.returncode == 0
    assert result.stdout.strip() == str(primary)


def test_accepts_explicit_alternate(tmp_path: Path) -> None:
    alternate = make_port(tmp_path / "alternate")

    result = run_resolver(tmp_path, RIGHT_TELEOP_PORT=str(alternate))

    assert result.returncode == 0
    assert result.stdout.strip() == str(alternate)


def test_explicit_missing_port_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    primary = make_port(tmp_path / "primary")

    result = run_resolver(
        tmp_path,
        RIGHT_TELEOP_PORT=str(missing),
        RIGHT_TELEOP_PRIMARY_PORT=str(primary),
    )

    assert result.returncode != 0
    assert "环境变量" in result.stderr
    assert str(missing) in result.stderr


def test_does_not_guess_another_ftdi(tmp_path: Path) -> None:
    alternate = make_port(tmp_path / "usb-FTDI_USB__-__Serial_Converter_FTAUMOPA-if00-port0")

    result = run_resolver(
        tmp_path,
        RIGHT_TELEOP_PRIMARY_PORT=str(tmp_path / "missing-primary"),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "拒绝自动选择其他 FTDI" in result.stderr
    assert str(alternate) in result.stderr
