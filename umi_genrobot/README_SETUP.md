# GenRobot DAS Finger Controller RV3 SDK environment

The physical robot-mounted unit was confirmed from its label as **DAS Finger
Controller RV3**. This directory therefore uses the one-camera Finger
Controller SDK, not the three-camera DAS Gripper Controller SDK.

The previously downloaded Gripper Controller SDK is retained under
`vendor/archived_wrong_model_gripper_controller/` and must not be run.

## Scope of phase 1

Phase 1 is **zero motion**. It verifies only:

- the folder-local Python 3.8 environment;
- importability of the official Finger Controller Python SDK;
- one serial device and one camera;
- a non-motion `MCUID` query;
- camera enumeration and advertised formats.

It does not start Franka, Robotiq, Finger motor control, encoder polling,
tactile polling, calibration, or teleoperation.

## Layout

```text
umi_genrobot/
├── .conda/
├── vendor/
│   ├── gen_finger_con_python_sdk_release/
│   └── archived_wrong_model_gripper_controller/
├── config/
├── scripts/doctor/
├── bridge/
├── logs/
├── requirements.lock.txt
└── SOURCE_PIN.txt
```

## Safe doctor sequence

```bash
cd /home/muka/frankateleop/umi_genrobot

./scripts/doctor/00_environment.sh
./scripts/doctor/01_usb_inventory.sh
.conda/bin/python scripts/doctor/02_sdk_import.py
```

The identity query requires an explicit, idle serial path:

```bash
SERIAL_PORT=/dev/serial/by-id/<confirmed-finger-controller-port> \
  ./scripts/doctor/03_query_identity.sh MCUID
```

See `config/OBSERVED_IDENTITY.md` for the live mapping.

## Forbidden in phase 1

Do not run:

```text
vendor/gen_finger_con_python_sdk_release/start_finger.py
1234
camerarc
any distance or sine-wave command
any Franka/Robotiq/teleop launcher
```

The vendor full launcher starts an encoder loop with an initial target of
0.0 m and then applies 0.05 m when no distance argument is supplied. Treat it
as a motion command, including when `--no-preview` is used.

## System and udev boundary

Ubuntu `v4l-utils` is the only system-level package installed. No ROS and no
vendor udev rules are installed. The vendor template uses physical USB paths
and mode `0666`; it must not be copied verbatim on this machine because the
existing Robotiq, homomorphic teleop adapter, and RealSense devices are live.

## Future bridge

The public Finger Controller SDK controls the robot-mounted Finger opening and
reads its one camera/tactile/encoder data. It does not provide the handheld DAS
Gripper live pose. That handheld pose source remains a separate integration
requirement.
