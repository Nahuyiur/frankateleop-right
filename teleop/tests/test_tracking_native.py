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


def policy(path,**extra):
    args=dict(joint_pos_current=torch.zeros(7),robot_model=Model(),
              **{k:torch.tensor(v) for k,v in BASELINE.items()})
    if path==0:p=TrackingHybridJointImpedanceControl(**args,record_id=123)
    else:
        poses=torch.eye(4,dtype=torch.float64).reshape(1,16).repeat(101,1)
        poses[:,12]=torch.arange(101)*.00001
        p=NativeTrackingHybridJointImpedanceControl(**args,path=path,poses=poses,record_id=123,
                                                   stiffness=torch.tensor([900.,900.,900.,20.,20.,20.]),**extra)
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


def test_c_requires_flange_frame_but_stopped_handoff_accepts_restoration():
    from types import SimpleNamespace
    import numpy as np
    import pytest
    from teleop.tracking_native import observed
    tool=np.eye(4);tool[:3,3]=[.044,0,.19]
    state=dict(path=2,log_failed=False,EE_T_K=np.eye(4).T.ravel().tolist())
    raw=dict(native_state_valid=True,state_timestamp_ns=time.time_ns(),native_F_T_EE=tool.T.ravel().tolist(),tracking_native=state)
    robot=SimpleNamespace(get_feedback_observations=lambda **_:raw)
    with pytest.raises(RuntimeError,match='not the flange'):observed(robot)
    state['EE_T_K']=np.linalg.inv(tool).T.ravel().tolist()
    assert observed(robot)[1]['path']==2
    state.update(path=3,EE_T_K=np.eye(4).T.ravel().tolist())
    assert observed(robot)[1]['path']==3


LIMIT=[2.5,2.5,2.5,2.5,1.5,1.,1.]


def integral_policy(path=1):
    return policy(path,integral_time_s=.5,integral_leak_s=5.,integral_limit_Nm=torch.tensor(LIMIT))


def run_integral(b,start,ms0,count,path=1,wall_offset_ms=0):
    out=None
    for k in range(count):
        ms=ms0+k
        out=b(state(path=path,ms=ms,wall=start+int((ms-ms0+wall_offset_ms)*1e6)+1000))
    return out


def test_integral_disabled_is_bitwise_unchanged_baseline():
    plain,disabled=policy(1),policy(1,integral_time_s=0.)
    s=state()
    with torch.no_grad():assert torch.equal(plain(s)['joint_torques'],disabled(s)['joint_torques'])


def test_integral_inactive_before_start_then_grows_bounded_and_decays():
    b,ref=integral_policy(),policy(1);now=time.time_ns();start=now+1000
    with torch.no_grad():
        for p in (b,ref):p.update(dict(start_wall_ns=torch.tensor([float(start)],dtype=torch.float64),
                                       lease_wall_ns=torch.tensor([float(start+60_000_000_000)],dtype=torch.float64)))
        # Waiting (path B active, not started): no integral contribution.
        idle=b(state(ms=9000,wall=now))['joint_torques']
        assert torch.allclose(idle,ref(state(ms=9000,wall=now))['joint_torques'])
        run_integral(b,start,10000,2000)            # 2 s of prelude with a constant q_d - q error
        s=state(ms=12000,wall=start+2_000_000_000+1000)
        delta=b(s)['joint_torques']-ref(s)['joint_torques']
        assert float(delta.abs().max())>.05, 'integral must act during the prelude/trajectory'
        assert torch.all(delta.abs()<=torch.tensor(LIMIT)+1e-5), 'per-joint torque limit'
        assert torch.all(torch.sign(delta)==torch.sign(state()['native_q_d']-state()['joint_positions'])), 'acts along the stiffness-weighted error'
        run_integral(b,start,12001,8000)            # saturate, then the endpoint (phase 3)
        s=state(ms=21500,wall=start+11_500_000_000)
        late=b(s)['joint_torques']-ref(s)['joint_torques']
        assert float(late.abs().max())<.01, 'integral must decay to ~0 at the stationary endpoint before A handoff'


def test_integral_never_reaches_c_or_path_a_hold():
    c=integral_policy(2)
    with torch.no_grad():assert torch.equal(c(state(path=2))['joint_torques'],torch.zeros(7))
    b,ref=integral_policy(),policy(1)
    with torch.no_grad():
        for k in range(500):s=state(path=0,ms=10000+k);x,y=b(s),ref(s)
    assert torch.allclose(x['joint_torques'],y['joint_torques'])


def test_integral_configuration_validation():
    import pytest
    from teleop.tracking_native import validate_integral
    assert validate_integral(None) is None
    ok=dict(time_s=.5,leak_s=5,limit_Nm=LIMIT)
    assert validate_integral(ok)['limit_Nm']==LIMIT
    for bad in (dict(ok,time_s=.1),dict(ok,leak_s=.4),dict(ok,limit_Nm=[4.]*7),dict(ok,limit_Nm=[1.]*6),
                dict(ok,extra=1),dict(time_s=.5,leak_s=5)):
        with pytest.raises(ValueError):validate_integral(bad)


FRICTION=[.3,.3,.3,.25,.2,.1,.1]


def test_friction_feedforward_follows_desired_velocity_only_in_b():
    import math
    b,ref=policy(1,friction_Nm=torch.tensor(FRICTION),friction_velocity_rad_s=.02),policy(1)
    with torch.no_grad():
        s=state();delta=b(s)['joint_torques']-ref(s)['joint_torques']
    expected=torch.tensor(FRICTION)*math.tanh(.1/.02)
    assert torch.allclose(delta,expected,atol=1e-5), 'tau_f = Fc tanh(dq_d/v0), measured velocity unused'
    with torch.no_grad():
        s=state(path=0);assert torch.allclose(b(s)['joint_torques'],ref(s)['joint_torques']), 'no friction term during the A hold'
        c=policy(2,friction_Nm=torch.tensor(FRICTION))
        assert torch.equal(c(state(path=2))['joint_torques'],torch.zeros(7))
        s=state();s['native_dq_d']=torch.zeros(7)
        assert torch.allclose(b(s)['joint_torques'],ref(s)['joint_torques']), 'zero desired velocity gives zero feedforward'


def test_friction_configuration_validation():
    import pytest
    from teleop.tracking_native import validate_friction
    assert validate_friction(None) is None
    ok=dict(coulomb_Nm=FRICTION,velocity_rad_s=.02)
    assert validate_friction(ok)['coulomb_Nm']==FRICTION
    for bad in (dict(ok,coulomb_Nm=[1.5]*7),dict(ok,coulomb_Nm=[-.1]*7),dict(ok,coulomb_Nm=[.1]*6),
                dict(ok,velocity_rad_s=.001),dict(ok,extra=1),dict(coulomb_Nm=FRICTION)):
        with pytest.raises(ValueError):validate_friction(bad)


def a_policy(**extra):
    p=TrackingHybridJointImpedanceControl(joint_pos_current=torch.zeros(7),robot_model=Model(),record_id=5,
        **{k:torch.tensor(v) for k,v in BASELINE.items()},**extra)
    stream=io.BytesIO();torch.jit.save(torch.jit.script(p),stream);stream.seek(0)
    return torch.jit.load(stream)


def a_step(p,ref,sequence,target,velocity):
    update=dict(joint_pos_desired=target,joint_vel_desired=velocity,target_sequence=torch.tensor([float(sequence)]))
    p.update(update);ref.update(update)
    s=state(path=0)
    return p(s)['joint_torques']-ref(s)['joint_torques']


def test_a_terms_off_is_bitwise_unchanged():
    plain,off=a_policy(),a_policy(integral_time_s=0.)
    with torch.no_grad():
        for p in (plain,off):p.update(dict(joint_pos_desired=torch.full((7,),.02),joint_vel_desired=torch.full((7,),.1),target_sequence=torch.ones(1)))
        s=state(path=0);assert torch.equal(plain(s)['joint_torques'],off(s)['joint_torques'])


def test_a_friction_follows_desired_velocity_and_fades_with_stale_targets():
    import math
    p,ref=a_policy(friction_Nm=torch.tensor(FRICTION),friction_velocity_rad_s=.02),a_policy()
    with torch.no_grad():
        delta=a_step(p,ref,1,torch.full((7,),.02),torch.full((7,),.1))
        assert torch.allclose(delta,torch.tensor(FRICTION)*math.tanh(.1/.02),atol=1e-5)
        for _ in range(60):s=state(path=0);delta=p(s)['joint_torques']-ref(s)['joint_torques']   # no new target for 60 ms
        assert float(delta.abs().max())<1e-6, 'stale target: desired velocity and friction term are zero'


def test_a_integral_waits_for_first_target_is_bounded_and_fades_when_idle():
    p,ref=a_policy(integral_time_s=.3,integral_leak_s=5.,integral_limit_Nm=torch.tensor(LIMIT)),a_policy()
    with torch.no_grad():
        for _ in range(500):s=state(path=0);x=p(s)['joint_torques'];y=ref(s)['joint_torques']
        assert torch.allclose(x,y), 'holding the prepared pose before any target: no integral'
        target=torch.full((7,),.02)
        for k in range(1,2001):
            delta=a_step(p,ref,(k+65)//66,target,torch.zeros(7))      # a new target every 66 ms
        assert float(delta.abs().max())>.05 and torch.all(delta.abs()<=torch.tensor(LIMIT)+1e-5)
        assert torch.all(torch.sign(delta)==torch.sign(target-state()['joint_positions']))
        for _ in range(900):s=state(path=0);delta=p(s)['joint_torques']-ref(s)['joint_torques']
        assert float(delta.abs().max())>.05, 'still active within the idle window'
        for _ in range(1500):s=state(path=0);delta=p(s)['joint_torques']-ref(s)['joint_torques']
        assert float(delta.abs().max())<.01, 'faded after targets stopped, before the handover'


def test_a_configure_carries_terms_and_rejects_zero_velocity_mode():
    from types import SimpleNamespace
    import pytest
    from teleop import tracking_impedance
    from teleop.tracking_gains import DEFAULT_VELOCITY_MODE
    reference=dict(sha256='a'*64,times_ns=[1,2],eef=[[.4,0,.3,0,0,0,1.]]*2)
    integral=dict(time_s=.3,leak_s=5.,limit_Nm=LIMIT);friction=dict(coulomb_Nm=FRICTION,velocity_rad_s=.02)
    robot=SimpleNamespace(control_mode='ee',_ik_mode='dik')
    out=tracking_impedance.configure(robot,BASELINE,reference,DEFAULT_VELOCITY_MODE,integral=integral,friction=friction)
    assert out['integral']==integral and out['friction']==friction and robot._tracking_terms==dict(integral=integral,friction=friction)
    robot=SimpleNamespace(control_mode='ee',_ik_mode='dik')
    assert 'integral' not in tracking_impedance.configure(robot,BASELINE,reference,DEFAULT_VELOCITY_MODE)
    with pytest.raises(ValueError):
        tracking_impedance.configure(SimpleNamespace(control_mode='ee',_ik_mode='dik'),BASELINE,reference,'zero',integral=integral)


if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_') and not fn.__code__.co_argcount:fn();print('PASS',name)
