"""Native target timing and A/B equation equivalence, without robot connections."""
import io
import time
import torch

from teleop.tracking_gains import BASELINE
from teleop.tracking_policy import TrackingHybridJointImpedanceControl, NativeTrackingHybridJointImpedanceControl


class Model(torch.nn.Module):
    def compute_jacobian(self,q:torch.Tensor):
        return torch.eye(6,7)*(1.+q[0])
    def inverse_dynamics(self,q:torch.Tensor,dq:torch.Tensor,ddq:torch.Tensor):
        return 3.+q+.25*dq+ddq


def policy(path):
    args=dict(joint_pos_current=torch.zeros(7),robot_model=Model(),
              **{k:torch.tensor(v) for k,v in BASELINE.items()})
    if path==0:p=TrackingHybridJointImpedanceControl(**args,record_id=123)
    else:
        poses=torch.eye(4,dtype=torch.float64).reshape(1,16).repeat(101,1)
        poses[:,12]=torch.arange(101)*.00001
        p=NativeTrackingHybridJointImpedanceControl(**args,path=path,poses=poses,record_id=123,
                                                   stiffness=torch.tensor([900.,900.,900.,20.,20.,20.]))
    stream=io.BytesIO();torch.jit.save(torch.jit.script(p),stream);stream.seek(0)
    return torch.jit.load(stream)


def state(path=1,ms=10000.,wall=None):
    wall=time.time_ns() if wall is None else wall
    return dict(joint_positions=torch.linspace(.001,.007,7),joint_velocities=torch.linspace(.01,.07,7),
                native_q_d=torch.full((7,),.02),native_dq_d=torch.full((7,),.1),
                native_robot_ms=torch.tensor([ms],dtype=torch.float64),
                tracking_control_path=torch.tensor([float(path)]),
                timestamp=torch.tensor([wall//10**9,wall%10**9],dtype=torch.int32))


def test_equal_inputs_equal_formula_and_trace():
    a,b=policy(0),policy(1);s=state()
    with torch.no_grad():
        a.update(dict(joint_pos_desired=s['native_q_d'],joint_vel_desired=s['native_dq_d'],target_sequence=torch.ones(1)))
        x,y=a(s),b(s)
    assert torch.equal(x['joint_torques'],y['joint_torques'])
    assert torch.equal(x['tracking_trace'][:21],y['tracking_trace'][:21])
    assert y['tracking_trace'][21]==1


def test_native_uses_current_state_qd_without_extra_cycle_shift():
    b=policy(1);s=state()
    with torch.no_grad():
        x=b(s);s['native_q_d']+=.01;y=b(s)
    assert torch.equal(y['tracking_trace'][:7],s['native_q_d'])
    assert not torch.equal(x['joint_torques'],y['joint_torques'])


def test_wait_prelude_timing_endpoint_and_owner_expiry():
    b=policy(1);now=time.time_ns();start=now+100_000_000
    with torch.no_grad():
        b.update(dict(start_wall_ns=torch.tensor([float(start)],dtype=torch.float64),
                      lease_wall_ns=torch.tensor([float(start+20_000_000_000)],dtype=torch.float64)))
        before=b(state(wall=now))['native_tracking_command'].clone()
        begun=b(state(ms=10000,wall=start+1000))['native_tracking_command'].clone()
        middle=b(state(ms=11500,wall=start+1_500_000_000))['native_tracking_command'].clone()
        sample=b(state(ms=13050,wall=start+3_050_000_000))['native_tracking_command'].clone()
        end=b(state(ms=14000,wall=start+4_000_000_000))['native_tracking_command'].clone()
        expired=b(state(ms=40000,wall=start+30_000_000_000))['native_tracking_command'].clone()
    assert before[27]==0 and before[3]==-1 and before[4]==0
    assert begun[27]==1 and begun[4]==0
    assert middle[27]==1 and abs(float(middle[4])-.5)<1e-12
    assert sample[27]==2 and sample[3]==50 and sample[4]==1
    assert end[27]==3 and end[3]==100
    assert expired[2]==2, 'Expired ownership must abort, not count as successful finish'


def test_c_never_submits_external_impedance_torque():
    c=policy(2)
    with torch.no_grad():
        assert torch.equal(c(state(path=2))['joint_torques'],torch.zeros(7))
        c.update(dict(finish=torch.ones(1)))
        assert c(state(path=2))['native_tracking_command'][2]==1


def test_mid_motion_finish_aborts_without_restoring_a(monkeypatch):
    from types import SimpleNamespace
    from teleop import tracking_native
    import pytest
    updates=[]
    robot=SimpleNamespace(robot=SimpleNamespace(update_current_policy=updates.append))
    monkeypatch.setattr(tracking_native,'active',lambda _:dict(record_id=123))
    for phase,velocity in ((2,0.),(3,.01)):
        monkeypatch.setattr(tracking_native,'observed',lambda _:(None,dict(phase=phase,dq_d=[velocity]*7)))
        with pytest.raises(RuntimeError,match='aborted before stationary endpoint'):
            tracking_native.finish(robot)
        assert updates[-1]['finish'].item()==2


def test_latched_fault_finish_does_not_mutate_policy(monkeypatch):
    from types import SimpleNamespace
    from teleop import tracking_native
    import pytest
    updates=[]
    robot=SimpleNamespace(robot=SimpleNamespace(update_current_policy=updates.append))
    monkeypatch.setattr(tracking_native,'active',lambda _:dict(record_id=123))
    def failed(_):raise RuntimeError('Latched native experiment/recorder fault')
    monkeypatch.setattr(tracking_native,'observed',failed)
    with pytest.raises(RuntimeError,match='Latched'):
        tracking_native.finish(robot)
    assert not updates


if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_') and not fn.__code__.co_argcount:fn();print('PASS',name)
