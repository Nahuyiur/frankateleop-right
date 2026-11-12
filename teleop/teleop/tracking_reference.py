"""Deterministic local reference for native experiments; never changes speed implicitly."""
import hashlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

VERSION='native-reference-50ms-v1'
PROBE_VERSION='native-short-probe-v1'
GUARDED_C_CLIENT_SHA256='3ea6032106db7160cdca38e68f61a839117d62a19c5bc2f62d97e5444839d91b'
GUARDED_C_REFERENCES=frozenset((
    '0e457b6fae976de9e15a63397006bc7a1447cbb7231ca1ed14e38d99666155ba',
    'ba2cad3eaf736dc8f50a90a3d77ca8947ce3b77d20517f279a888029212c1637'))
LIMITS=dict(translation_velocity=2.4,translation_acceleration=7.2,translation_jerk=3600.,
            rotation_velocity=2.,rotation_acceleration=13.6,rotation_jerk=6800.)


def check_native_mode(path,poses,metadata=None):
    # After workspace clearance and probes 25/26, allow only the approved
    # reference, with configure() additionally requiring the guarded client.
    a=np.asarray(poses)
    if (metadata or {}).get('version')==PROBE_VERSION:
        tool=matrix(metadata['F_T_EE'])
        origin=matrix(a[0])@np.linalg.inv(tool)
        expected,_,_=probe(np.r_[origin[:3,3],Rotation.from_matrix(origin[:3,:3]).as_quat()],metadata['F_T_EE'])
        if a.shape!=expected.shape or not np.allclose(a,expected,rtol=0,atol=1e-10):
            raise ValueError('Short probe must exactly match the fixed 8 s, 5 mm / 1 degree profile')
        return
    if path=='C' and not np.array_equal(a,np.broadcast_to(a[0],a.shape)):
        meta=metadata or {}
        if (meta.get('version')==VERSION and meta.get('speed')==.5 and len(a)==40689
                and hashlib.sha256(a.astype('<f8').tobytes()).hexdigest() in GUARDED_C_REFERENCES):
            return GUARDED_C_CLIENT_SHA256
        raise ValueError('Dynamic C disabled after run 22 table collision; independent validation required')


def probe(flange_pose,tool):
    """Small out-and-back diagnostic, separate from the fixed-trajectory comparison."""
    t=np.arange(8001)*.001
    u=np.minimum(t,8.-t)/4.
    weight=10*u**3-15*u**4+6*u**5
    origin=np.asarray(flange_pose,dtype=float)
    rotations=Rotation.from_quat(origin[3:])*Rotation.from_rotvec(
        np.c_[np.zeros(len(t)),weight*np.pi/180.,np.zeros(len(t))])
    flange=np.tile(np.eye(4),(len(t),1,1))
    flange[:,:3,:3]=rotations.as_matrix();flange[:,:3,3]=origin[:3]
    flange[:,2,3]+=.005*weight
    poses=(flange@matrix(tool)).transpose(0,2,1).reshape(-1,16).astype('<f8')
    return poses,np.c_[flange[:,:3,3],rotations.as_quat()],dict(
        version=PROBE_VERSION,speed=1.,sample_period_s=.001,duration_s=8.,
        processed_sha256=hashlib.sha256(poses.tobytes()).hexdigest(),F_T_EE=list(tool),
        peaks=check(poses),diagnostic_only=True,
        profile='4 s quintic outbound + 4 s return; flange base Z +5 mm and local Y +1 degree; stationary endpoints')


def matrix(values):
    a=np.asarray(values,dtype=float).reshape(4,4,order='F')
    if (not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-8)
            or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-6)
            or abs(np.linalg.det(a[:3,:3])-1)>1e-6):
        raise ValueError('Invalid rigid transform')
    return a


def check(poses):
    a=np.asarray(poses,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=16 or not 2<=len(a)<=60001 or not np.isfinite(a).all():
        raise ValueError('Native reference requires 2–60001 finite 1 ms poses')
    transforms=a.reshape(-1,4,4).transpose(0,2,1)
    r=transforms[:,:3,:3]
    if (not np.allclose(transforms[:,3,:],[0,0,0,1],atol=1e-8)
            or not np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-6)
            or not np.allclose(np.linalg.det(r),1.,atol=1e-6)):
        raise ValueError('Native reference contains an invalid rotation/transform')
    # Include constant samples at both ends to check transitions into/out of hold.
    pos=np.pad(transforms[:,:3,3],((3,3),(0,0)),mode='edge')
    rotations=Rotation.from_matrix(np.concatenate((r[:1].repeat(3,axis=0),r,r[-1:].repeat(3,axis=0))))
    velocities=np.c_[np.diff(pos,axis=0)/.001,(rotations[1:]*rotations[:-1].inv()).as_rotvec()/.001]
    peaks={}
    for order,name in enumerate(('velocity','acceleration','jerk')):
        values=velocities if order==0 else np.diff(velocities,n=order,axis=0)/.001**order
        for axis,indices in [('translation',slice(0,3)),('rotation',slice(3,6))]:
            peaks[axis+'_'+name]=float(np.max(np.linalg.norm(values[:,indices],axis=1)))
    failed={k:v for k,v in peaks.items() if v>LIMITS[k]}
    if failed:
        raise ValueError('Reference exceeds native experiment limits; speed was NOT changed: '+str(failed))
    return peaks


def build(reference,start_ns,end_ns,tool,speed=1.):
    if speed not in (1.,.5):raise ValueError('Only explicit 1× and separate 0.5× groups are supported')
    t=(np.asarray(reference['times_ns'],dtype=np.int64)-int(start_ns))/1e9
    raw=np.asarray(reference['eef'],dtype=np.float64)
    if raw.shape!=(len(t),7) or len(t)<2 or np.any(np.diff(t)<=0) or not np.isfinite(raw).all():
        raise ValueError('Invalid raw reference')
    duration=(int(end_ns)-int(start_ns))/1e9
    if not 0<duration/speed<=60 or t[0]<0 or t[-1]>duration:raise ValueError('Invalid reference time range')
    rotations=Rotation.from_quat(raw[:,3:])
    rv=(rotations[0].inv()*rotations).as_rotvec()
    if np.max(np.linalg.norm(rv,axis=1))>2.8:raise ValueError('Reference crosses the supported rotation-vector chart')
    source_grid=np.linspace(0,duration,round(duration*1000)+1)
    raw_curve=CubicSpline(t,np.c_[raw[:,:3],rv],bc_type=((1,np.zeros(6)),(1,np.zeros(6))))
    source_values=raw_curve(np.clip(source_grid,t[0],t[-1]))
    smooth=gaussian_filter1d(source_values,50.,axis=0,mode='nearest')
    # Preserve exact endpoints and join the stationary prelude/endpoint hold with
    # zero first and second derivatives. No causal delay is introduced here.
    for reverse in (False,True):
        distance=(duration-source_grid) if reverse else source_grid
        weight=np.clip(distance/.2,0,1)
        weight=10*weight**3-15*weight**4+6*weight**5
        anchor=source_values[-1] if reverse else source_values[0]
        smooth=anchor+(smooth-anchor)*weight[:,None]
    output_t=np.arange(round(duration/speed*1000)+1)*.001
    values=CubicSpline(source_grid,smooth)(np.minimum(output_t*speed,duration))
    flange=np.tile(np.eye(4),(len(values),1,1))
    flange[:,:3,:3]=(rotations[0]*Rotation.from_rotvec(values[:,3:])).as_matrix()
    flange[:,:3,3]=values[:,:3]
    ee=flange@matrix(tool)
    poses=ee.transpose(0,2,1).reshape(-1,16).astype('<f8')
    peaks=check(poses)
    direct=raw_curve(np.clip(output_t*speed,t[0],t[-1]))
    direct_rot=rotations[0]*Rotation.from_rotvec(direct[:,3:])
    actual_rot=Rotation.from_matrix(flange[:,:3,:3])
    deviation=dict(position_max_mm=float(np.max(np.linalg.norm(values[:,:3]-direct[:,:3],axis=1))*1000),
                   rotation_max_deg=float(np.max((actual_rot*direct_rot.inv()).magnitude())*180/np.pi))
    metadata=dict(version=VERSION,speed=speed,sample_period_s=.001,duration_s=float(output_t[-1]),
                  source_sha256=reference['sha256'],processed_sha256=hashlib.sha256(poses.tobytes()).hexdigest(),
                  limits=LIMITS,peaks=peaks,processing_deviation=deviation,F_T_EE=list(tool),
                  smoothing='zero-phase Gaussian sigma 50 ms in original time; 200 ms C2 endpoint join',
                  native_processing='libfranka 100 Hz lowpass + rate limiter; actual O_T_EE_c recorded separately')
    flange_poses=np.c_[flange[:,:3,3],actual_rot.as_quat()]
    return poses,flange_poses,metadata
