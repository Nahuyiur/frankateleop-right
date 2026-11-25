"""Offline regression fixture: warmup must restore parameters and counters."""
from typing import Dict
from pathlib import Path
import torch
from polymetis.robot_interface import ParamDictContainer
import torchcontrol as toco

class StatefulPolicy(toco.PolicyModule):
    def __init__(self):
        super().__init__()
        self.target = torch.nn.Parameter(torch.ones(7) * 3.)
        self.ticks = 0
        self.updates = 0

    @torch.jit.export
    def update(self, values: Dict[str, torch.Tensor]) -> None:
        for key in values.keys():
            self._param_dict[key].data.copy_(values[key])
        self.updates += 1

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.ticks += 1
        return {'joint_torques': self.target + self.ticks + self.updates * 100.}

out=Path(__file__).resolve().parent
torch.jit.save(torch.jit.script(StatefulPolicy()),str(out/'stateful.pt'))
torch.jit.save(torch.jit.script(ParamDictContainer({'target':torch.ones(7)*9.})),str(out/'stateful_params.pt'))
