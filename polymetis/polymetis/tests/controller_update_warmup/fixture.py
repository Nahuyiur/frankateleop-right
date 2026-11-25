"""Export a current native policy fixture using read-only metadata and feedback."""
import json
from pathlib import Path
import torch
import torchcontrol as toco
from polymetis import RobotInterface
from polymetis.robot_interface import ParamDictContainer

out = Path(__file__).resolve().parent
robot = RobotInterface(ip_address='127.0.0.1')
q = robot.get_joint_positions()
policy = toco.policies.HybridJointImpedanceControl(
    joint_pos_current=q, Kq=robot.Kq_default, Kqd=robot.Kqd_default,
    Kx=robot.Kx_default, Kxd=robot.Kxd_default,
    robot_model=robot.robot_model, ignore_gravity=robot.use_grav_comp)
torch.jit.save(torch.jit.script(policy), str(out/'policy.pt'))
torch.jit.save(torch.jit.script(ParamDictContainer({'joint_pos_desired': q})), str(out/'params.pt'))
(out/'q.txt').write_text(' '.join(str(x) for x in q.tolist()))
(out/'fixture.json').write_text(json.dumps(dict(
    policy=type(policy).__name__, q=q.tolist(), hz=robot.hz,
    gains={k:getattr(robot,k+'_default').tolist() for k in ['Kq','Kqd','Kx','Kxd']},
    mutation=False), indent=2))
print((out/'fixture.json').read_text())
