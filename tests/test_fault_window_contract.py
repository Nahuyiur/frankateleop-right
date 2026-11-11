"""Pure fake contracts against staged source. Never constructs a robot interface."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace, MethodType
import textwrap
import unittest
from unittest.mock import patch
import numpy as np
import torch
import grpc
from polymetis_pb2 import Empty, LogInterval

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / 'polymetis/polymetis/python/polymetis'
spec = importlib.util.spec_from_file_location('test_local_timing', PY/'_mozi_timing.py')
timing = importlib.util.module_from_spec(spec);spec.loader.exec_module(timing)

def method(path, name):
    source = path.read_text(); nodes = [n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes)==1
    node = nodes[0]
    body = textwrap.dedent(''.join(source.splitlines(True)[node.lineno-1:node.end_lineno]))
    scope = dict(torch=torch,np=np,grpc=grpc,span=timing.span,note=timing.note,
                 EMPTY=Empty(),ParamDictContainer=lambda x:x,time=__import__('time'),
                 log=__import__('logging').getLogger('fake-contract'))
    exec('from __future__ import annotations\n'+body,scope)
    return scope[name]

class RpcFailure(grpc.RpcError):
    def details(self):return 'fake RPC failed'

class BrainContracts(unittest.TestCase):
    def test_hold_anchor_requires_same_live_episode_and_valid_goal(self):
        fn=method(ROOT/'teleop/teleop/robots/fr3.py','_cartesian_hold_anchor')
        calls=[]
        interval=SimpleNamespace(start=12,end=-1)
        def query(empty,timeout):calls.append(timeout);return interval
        robot=SimpleNamespace(_teleop_accepted_joint_goal=[.1]*7,_teleop_goal_episode=12,
            grpc_connection=SimpleNamespace(GetEpisodeInterval=query),
            robot_model=SimpleNamespace(forward_kinematics=lambda q:(q[:3],torch.tensor([0.,0.,0.,1.]))))
        arm=SimpleNamespace(robot=robot,control_mode='ee')
        result=fn(arm);self.assertEqual(result['ee_hold_policy_episode'],12)
        np.testing.assert_allclose(result['ee_hold_pose_euler'][:3],[.1]*3)
        self.assertEqual(calls,[.25])
        interval.start=13;self.assertEqual(fn(arm),{})
        interval.start=12;interval.end=100;self.assertEqual(fn(arm),{})
        interval.end=-1;robot._teleop_accepted_joint_goal=[float('nan')]*7;self.assertEqual(fn(arm),{})
        arm.control_mode='joint';calls.clear();self.assertEqual(fn(arm),{});self.assertEqual(calls,[])

    def test_failed_episode_query_does_not_reuse_hold_goal(self):
        fn=method(ROOT/'teleop/teleop/robots/fr3.py','_cartesian_hold_anchor')
        robot=SimpleNamespace(_teleop_accepted_joint_goal=[.1]*7,_teleop_goal_episode=12,
             grpc_connection=SimpleNamespace(GetEpisodeInterval=lambda *a,**k:(_ for _ in ()).throw(RpcFailure())))
        self.assertEqual(fn(SimpleNamespace(robot=robot,control_mode='ee')),{})

    def test_joint_goal_cache_changes_only_after_successful_update(self):
        fn=method(PY/'robot_interface.py','update_current_policy')
        driver=SimpleNamespace(UpdateController=lambda stream:LogInterval(start=88),
                               GetEpisodeInterval=lambda empty:LogInterval(start=12,end=-1))
        robot=SimpleNamespace(grpc_connection=driver,_get_msg_generator=lambda value:lambda:iter(()),
                              _teleop_accepted_joint_goal=[0.]*7,_teleop_goal_episode=7)
        with patch('torch.jit.script',side_effect=lambda value:value):
            index=fn(robot,{'joint_pos_desired':torch.tensor([.1]*7)})
        self.assertEqual(index,76);self.assertEqual(robot._teleop_goal_episode,12)
        np.testing.assert_allclose(robot._teleop_accepted_joint_goal,[.1]*7)
        driver.UpdateController=lambda stream:(_ for _ in ()).throw(RpcFailure())
        with patch('torch.jit.script',side_effect=lambda value:value),self.assertRaises(grpc.RpcError):
            fn(robot,{'joint_pos_desired':torch.tensor([.9]*7)})
        np.testing.assert_allclose(robot._teleop_accepted_joint_goal,[.1]*7)

    def test_start_policy_replaces_goal_cache_and_failed_start_preserves_it(self):
        fn=method(PY/'robot_interface.py','send_torch_policy')
        driver=SimpleNamespace(SetController=lambda stream:LogInterval(start=99))
        robot=SimpleNamespace(grpc_connection=driver,_get_msg_generator=lambda value:lambda:iter(()),
                              _teleop_accepted_joint_goal=[0.]*7,_teleop_goal_episode=7)
        policy=SimpleNamespace(joint_pos_desired=torch.tensor([.2]*7))
        with patch('torch.jit.script',side_effect=lambda value:value):fn(robot,policy,blocking=False)
        self.assertEqual(robot._teleop_goal_episode,99)
        np.testing.assert_allclose(robot._teleop_accepted_joint_goal,[.2]*7)
        driver.SetController=lambda stream:(_ for _ in ()).throw(RpcFailure())
        with patch('torch.jit.script',side_effect=lambda value:value),self.assertRaises(grpc.RpcError):
            fn(robot,SimpleNamespace(joint_pos_desired=torch.tensor([.9]*7)),blocking=False)
        np.testing.assert_allclose(robot._teleop_accepted_joint_goal,[.2]*7)

    def test_request_details_are_bounded_json_and_keep_producer_timestamps(self):
        token=timing.begin_request('fake-command')
        timing.note('q',[1.,float('nan'),float('inf')])
        timing.note('text','x'*1000)
        for i in range(40):timing.note('value'+str(i),i)
        with timing.span('fake_policy_jit'):pass
        result=timing.end_request(token)
        self.assertEqual(len(result['details']),32)
        self.assertEqual(result['details']['q'],[1.,None,None])
        self.assertEqual(len(result['details']['text']),512)
        self.assertGreater(result['overflow'],0)
        self.assertGreater(result['start_monotonic_ns'],0)
        self.assertGreater(result['start_wall_ns'],0)
        json.dumps(result,allow_nan=False)

if __name__=='__main__':unittest.main()
