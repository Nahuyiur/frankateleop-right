"""Existing hybrid impedance with an explicit, expiring desired joint velocity."""
from typing import Dict
import math
import time

import torch
from torchcontrol.policies import HybridJointImpedanceControl

from teleop.tracking_gains import VELOCITY_REFERENCE


class TrackingHybridJointImpedanceControl(HybridJointImpedanceControl):
    def __init__(self, joint_pos_current, Kq, Kqd, Kx, Kxd, robot_model, ignore_gravity=True, record_id=0):
        super().__init__(joint_pos_current,Kq,Kqd,Kx,Kxd,robot_model,ignore_gravity)
        self.joint_vel_desired = torch.nn.Parameter(torch.zeros_like(self.joint_pos_desired))
        self.target_sequence = torch.nn.Parameter(torch.zeros(1))
        self.previous_sequence = -1.
        self.target_age_ticks = 0
        self.fade_ticks = int(1000*VELOCITY_REFERENCE['stale_fade_start_s'])
        self.zero_ticks = int(1000*VELOCITY_REFERENCE['stale_zero_s'])
        self.max_velocity = float(VELOCITY_REFERENCE['max_abs_rad_s'])
        self.native_tracking_command = torch.zeros(28,dtype=torch.float64)
        self.native_tracking_command[1] = float(record_id)
        self.native_tracking_command[3] = -1.

    def forward(self, state_dict: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        sequence = float(self.target_sequence[0])
        if sequence != self.previous_sequence:
            self.previous_sequence = sequence
            self.target_age_ticks = 0
        else:
            self.target_age_ticks = min(self.target_age_ticks+1,self.zero_ticks)
        freshness = max(0.,min(1.,float(self.zero_ticks-self.target_age_ticks)/(self.zero_ticks-self.fade_ticks)))
        velocity = torch.clamp(self.joint_vel_desired,-self.max_velocity,self.max_velocity)*freshness
        q,dq = state_dict['joint_positions'],state_dict['joint_velocities']
        feedback = self.joint_pd(q,dq,self.joint_pos_desired,velocity,self.robot_model.compute_jacobian(q))
        feedforward = self.invdyn(q,dq,torch.zeros_like(q))
        trace = torch.cat((self.joint_pos_desired, velocity, feedforward, torch.zeros(1)))
        return {'joint_torques':feedback+feedforward, 'tracking_trace':trace,
                'native_tracking_command':self.native_tracking_command}


class NativeTrackingHybridJointImpedanceControl(HybridJointImpedanceControl):
    """B: identical hybrid PD/model/ID, with native q_d and dq_d inputs.

    During the explicit stopped handoff, path A holds the initialization pose.
    The native client requires this policy's trace before entering B.
    """
    def __init__(self, joint_pos_current, Kq, Kqd, Kx, Kxd, robot_model,
                 poses, path, record_id, stiffness, ignore_gravity=True,
                 integral_time_s=0., integral_leak_s=5., integral_limit_Nm=None,
                 friction_Nm=None, friction_velocity_rad_s=.02):
        super().__init__(joint_pos_current,Kq,Kqd,Kx,Kxd,robot_model,ignore_gravity)
        # Optional B-only quasi-static correction (P1). Disabled (0) by default:
        #   tau_i = Kp(q) z,  dz/dt = (q_d - q)/Ti - z/T_leak,  |tau_i| <= limit (uniform scaling),
        # active in phases 1-2 only; decays with 0.2 s after the endpoint so A receives no step.
        self.integral_time_s = float(integral_time_s)
        self.integral_leak_s = float(integral_leak_s)
        self.integral_limit = (torch.zeros(7,dtype=torch.float32) if integral_limit_Nm is None
                               else torch.as_tensor(integral_limit_Nm,dtype=torch.float32).clone())
        self.integral_state = torch.zeros(7,dtype=torch.float32)
        self.integral_last_ms = -1.
        self.integral_phase = 0 # phase of the previous cycle (computed after the torque)
        self.integral_last_torque = torch.zeros(7,dtype=torch.float32)
        # Optional B-only Coulomb friction feedforward on the DESIRED velocity (no measured-velocity feedback):
        #   tau_f = Fc * tanh(dq_d / v0). Zero when dq_ref is zero (waiting, path A hold, C).
        self.friction_enabled = friction_Nm is not None
        self.friction = (torch.zeros(7,dtype=torch.float32) if friction_Nm is None
                         else torch.as_tensor(friction_Nm,dtype=torch.float32).clone())
        self.friction_velocity = float(friction_velocity_rad_s)

        # Complete immutable 1 kHz reference is transferred before handoff.
        self.poses = poses.to(dtype=torch.float64)
        self.path = int(path)
        self.command = torch.zeros(28,dtype=torch.float64)
        self.command[0] = float(path)
        self.command[1] = float(record_id)
        self.command[5:11] = stiffness.to(dtype=torch.float64)
        self.start_wall_ns = torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
        self.finish = torch.nn.Parameter(torch.zeros(1))
        # Bounded installation window; the coordinator shortens this to 3 s once both policies are ready.
        self.lease_wall_ns = torch.nn.Parameter(torch.tensor([float(time.time_ns()+20_000_000_000)],dtype=torch.float64))
        self.started_robot_ms = -1.

    def integral_torque(self, q:torch.Tensor, q_ref:torch.Tensor, jacobian:torch.Tensor,
                        active:int, ms:float) -> torch.Tensor:
        dt = 0.
        if self.integral_last_ms >= 0.:
            dt = min(max((ms-self.integral_last_ms)*1.e-3,0.),.005)
        self.integral_last_ms = ms
        z = self.integral_state
        if active == 1 and (self.integral_phase == 1 or self.integral_phase == 2):
            z = z+dt*((q_ref-q).to(dtype=torch.float32)/self.integral_time_s-z/self.integral_leak_s)
        else:
            z = z*math.exp(-dt/.2)
        stiffness = self.joint_pd.Kq+jacobian.T@self.joint_pd.Kx@jacobian
        tau = stiffness@z
        ratio = float(torch.max(torch.abs(tau)/self.integral_limit))
        if ratio > 1.:
            z = z/ratio
            tau = tau/ratio
        self.integral_state = z
        self.integral_last_torque = tau
        return tau

    def forward(self, state_dict: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        q,dq = state_dict['joint_positions'],state_dict['joint_velocities']
        q_ref,dq_ref = self.joint_pos_desired,torch.zeros_like(q)
        active = int(state_dict['tracking_control_path'][0])
        if active == 1:
            q_ref,dq_ref = state_dict['native_q_d'],state_dict['native_dq_d']
        jacobian = self.robot_model.compute_jacobian(q)
        feedback = self.joint_pd(q,dq,q_ref,dq_ref,jacobian)
        feedforward = self.invdyn(q,dq,torch.zeros_like(q))
        torque = feedback+feedforward
        if self.friction_enabled and active == 1:
            torque = torque+self.friction*torch.tanh(dq_ref.to(dtype=torch.float32)/self.friction_velocity)
        if self.integral_time_s > 0.:
            torque = torque+self.integral_torque(q,q_ref,jacobian,active,float(state_dict['native_robot_ms'][0]))
        if active == 2: # Computed external torques are never sent in C.
            torque = torch.zeros_like(q)
        trace = torch.cat((q_ref,dq_ref,feedforward,torch.full((1,),float(self.path))))
        stamp = state_dict['timestamp'].to(dtype=torch.float64)
        now = float(stamp[0])*1.e9+float(stamp[1])
        start = float(self.start_wall_ns[0])
        ms = float(state_dict['native_robot_ms'][0])
        if active == self.path and start > 0. and now >= start and self.started_robot_ms < 0.:
            self.started_robot_ms = ms
        index,blend,phase = 0,0.,0
        if self.started_robot_ms >= 0.:
            elapsed = max(0.,ms-self.started_robot_ms)
            if elapsed < 3000.:
                s = elapsed/3000.
                blend = 10.*s**3-15.*s**4+6.*s**5
                phase = 1
            else:
                index = min(int(elapsed)-3000,self.poses.size(0)-1)
                blend = 1.
                phase = 2 if int(elapsed)-3000 < self.poses.size(0)-1 else 3
        self.command[2] = float(self.finish[0]) if now <= float(self.lease_wall_ns[0]) else 2.
        self.command[3] = float(index) if phase >= 2 else -1.
        self.command[4] = blend
        self.command[11:27] = self.poses[index]
        self.command[27] = float(phase)
        self.integral_phase = phase
        return {'joint_torques':torque,'tracking_trace':trace,
                'native_tracking_command':self.command}
