"""Offline native diagnostic decoder. Standard library only; no robot access."""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import sys

SCALARS64 = ['sequence','mono','wall','robot_ms','period_us','rpc_begin','rpc_end',
             'state_wall','torque_wall','current_errors','last_motion_errors','dropped']
ARRAYS = {**{key:7 for key in ['q','dq','q_d','dq_d','tau','tau_d','tau_ext',
           'policy_torque','safety_torque','returned_torque']},'O_T_EE':16,
           'O_F_ext_hat_K':6,'joint_contact':7,'joint_collision':7,
           'cartesian_contact':6,'cartesian_collision':6}

class Sample(C.Structure):
    _fields_ = [(key,C.c_uint64) for key in SCALARS64] + [
        (key,C.c_uint32) for key in ['valid','source','robot_mode']] + [
        ('success_rate',C.c_double)] + [(key,C.c_double*size) for key,size in ARRAYS.items()]

def errors(mask,names):
    return [names[i] if i<len(names) else 'unmapped_native_error_bit_'+str(i)
            for i in range(64) if mask & (1<<i)]

def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,list):return [clean(v) for v in value]
    return value

def decode(manifest,output,policy=None):
    meta=json.loads(manifest.read_text())
    if sys.byteorder!='little' or C.sizeof(Sample)!=meta['sample_bytes'] or meta['schema']!=1:
        raise ValueError('Unsupported native schema/ABI; refusing a guessed decode')
    binary=manifest.with_suffix('.bin')
    if binary.stat().st_size!=meta['sample_bytes']*meta['sample_count']:
        raise ValueError('Binary is incomplete or does not match its completed manifest')
    names=meta['metadata'].get('error_names',[])
    output.mkdir(parents=True,exist_ok=True)
    peaks={key:[0.]*7 for key in ['dq','tau','tau_d','policy_torque','returned_torque']}
    summary={'manifest':str(manifest),'sample_count':meta['sample_count'],
             'truncated':meta['truncated'],'dropped':meta['drop_end']-meta['drop_begin'],
             'available_pre_s':meta.get('available_pre_ns',0)/1e9,
             'native_errors':[],'gaps':[],'peak_abs':peaks,'first_forward_matches':[]}
    native_names=set();state_timestamps={};last=None;seen_rates=[]
    with binary.open('rb') as source,(output/'samples.jsonl').open('w') as target:
        for _ in range(meta['sample_count']):
            s=Sample.from_buffer_copy(source.read(C.sizeof(Sample)))
            row={key:int(getattr(s,key)) for key in SCALARS64}
            row.update(valid=int(s.valid),source='callback' if s.source==0 else 'exception_log',
                       robot_mode=int(s.robot_mode),success_rate=clean(s.success_rate))
            for key in ARRAYS:row[key]=clean(list(getattr(s,key)))
            if not (s.valid&2):
                for key in ['policy_torque','safety_torque','returned_torque']:row[key]=None
            row['current_error_names']=errors(s.current_errors,names)
            row['last_motion_error_names']=errors(s.last_motion_errors,names)
            native_names.update(row['current_error_names']);native_names.update(row['last_motion_error_names'])
            # Do not use exception-log collection time as a new feedback sample.
            if s.source==0:
                state_timestamps[int(s.state_wall)]=int(s.sequence)
                seen_rates.append(s.success_rate)
                if last is not None and (s.robot_ms-last.robot_ms>5 or s.sequence-last.sequence>1):
                    summary['gaps'].append({'after_sequence':int(last.sequence),'next_sequence':int(s.sequence),
                        'robot_source_gap_ms':int(s.robot_ms-last.robot_ms),
                        'callback_source_gap_ms':(s.mono-last.mono)/1e6})
                last=s
                for key in peaks:
                    if row[key] is not None:
                        peaks[key]=[max(old,abs(v)) if v is not None else old for old,v in zip(peaks[key],row[key])]
            target.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    summary['native_errors']=sorted(native_names)
    finite_rates=[v for v in seen_rates if math.isfinite(v)]
    summary['min_callback_success_rate']=min(finite_rates) if finite_rates else None
    if policy:
        with policy.open() as source:
            for line in source:
                event=json.loads(line)
                stamp=event['state_source_wall_ns']
                if event['kind']==3 and stamp in state_timestamps:
                    summary['first_forward_matches'].append({**event,'native_sequence':state_timestamps[stamp]})
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',type=Path)
    args=parser.parse_args()
    result=decode(args.manifest,args.output,args.policy)
    print(json.dumps({key:result[key] for key in ['sample_count','native_errors','truncated','dropped']},ensure_ascii=False))
