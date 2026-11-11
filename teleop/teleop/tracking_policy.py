"""Existing hybrid impedance with an explicit, expiring desired joint velocity."""
from typing import Dict

import torch
from torchcontrol.policies import HybridJointImpedanceControl

from teleop.tracking_gains import VELOCITY_REFERENCE


class TrackingHybridJointImpedanceControl(HybridJointImpedanceControl):
    def __init__(self, joint_pos_current, Kq, Kqd, Kx, Kxd, robot_model, ignore_gravity=True):
        super().__init__(joint_pos_current,Kq,Kqd,Kx,Kxd,robot_model,ignore_gravity)
        self.joint_vel_desired = torch.nn.Parameter(torch.zeros_like(self.joint_pos_desired))
        self.target_sequence = torch.nn.Parameter(torch.zeros(1))
        self.previous_sequence = -1.
        self.target_age_ticks = 0
        self.fade_ticks = int(1000*VELOCITY_REFERENCE['stale_fade_start_s'])
        self.zero_ticks = int(1000*VELOCITY_REFERENCE['stale_zero_s'])
        self.max_velocity = float(VELOCITY_REFERENCE['max_abs_rad_s'])

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
        return {'joint_torques':feedback+feedforward}
