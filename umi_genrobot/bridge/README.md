# Bridge status

## Robot-mounted DAS Finger actuator bridge

The robot-mounted DAS Finger Controller RV3 is exposed to the existing
Polymetis gripper API by `das_finger_polymetis_server.py`.

- Serial device: `/dev/ttyFingerLeft` (CH340, 921600 baud)
- Camera device: `/dev/finger_camera_left`
- Polymetis server: the port supplied by `2_launch_gripper.sh` (50052 in the
  single-left-arm GUI)
- Driver selection: `LEFT_GRIPPER_DRIVER=auto` chooses `das_finger` when
  `/dev/ttyFingerLeft` exists; set it to `robotiq` to use the legacy driver.
- Startup safety: opening stage 2 does not send a distance packet. The bridge
  starts its 30 Hz command/encoder loop only after the first explicit Polymetis
  `Goto` request.

The vendor protocol exposes target opening distance and encoder feedback, but
does not expose the Polymetis speed or force controls. Those two command fields
are therefore accepted for API compatibility and are not sent to the device.

## Handheld live-pose bridge

No handheld live-pose bridge is active. The handheld live-pose source still
needs to be identified and verified separately.

Planned control contract after the handheld DAS Gripper live pose interface is
available:

```text
toggle ON:
  T_U0 = current handheld pose
  T_R0 = current Franka end-effector pose

each sample:
  delta_U = inverse(T_U0) @ T_U
  delta_R = X @ delta_U @ inverse(X)
  T_R_des = T_R0 @ delta_R

toggle OFF:
  stop publishing new targets
```

The public SDK and Matrix documentation currently establish recorded MCAP pose
topics, not a public live handheld pose stream. Do not connect this directory
to a real Franka controller until that interface and its coordinate/time
contract are verified.
