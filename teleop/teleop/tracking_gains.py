"""Shared experiment contract; identical copy is installed on both arm backends."""
import math

BASELINE = dict(Kq=[40.,30.,50.,25.,35.,25.,10.], Kqd=[4.,6.,5.,5.,3.,2.,1.],
                Kx=[750.,750.,750.,15.,15.,15.], Kxd=[37.,37.,37.,2.,2.,2.])
FORMULA = ('τ_policy = Kp(q)(q_des − q) + Kd(q)(dq_des − dq) + τ_ff; '
           'Kp(q) = diag(Kq) + Jᵀdiag(Kx)J; Kd(q) = diag(Kqd) + Jᵀdiag(Kxd)J; '
           'τ_ff = ID(q,dq,0) − ID(q,0,0)')
LIMIT_FACTORS = [.5, 1.5]
VELOCITY_REFERENCE = dict(version=1, mode='validated_trajectory_joint_difference',
    filter_tau_s=.02, max_abs_rad_s=1., max_accel_rad_s2=10.,
    stale_fade_start_s=.03, stale_zero_s=.05,
    first_target='zero', timebase='fixed_trajectory_source_ns_at_1x')
DEFAULT_VELOCITY_MODE = VELOCITY_REFERENCE['mode']


def control_law(mode):
    if mode == 'zero':
        return dict(id='hybrid-zero-v1', mode=mode, label='原公式 · 目标速度为 0',
                    formula=FORMULA.replace('dq_des − dq', '0 − dq'),
                    velocity_formula='dq_des = 0', velocity_reference=dict(version=1, mode=mode))
    if mode == DEFAULT_VELOCITY_MODE:
        return dict(id='hybrid-trajectory-difference-v1', mode=mode, label='新公式 · 轨迹差分目标速度',
                    formula=FORMULA,
                    velocity_formula='dq_raw[k] = (q_des[k] − q_des[k−1]) / (t[k] − t[k−1]); '
                        'dq_des = rate_limit(lowpass(clip(dq_raw))); stale/HOLD → 0',
                    velocity_reference=dict(VELOCITY_REFERENCE))
    raise ValueError('目标速度只支持 zero 或 validated_trajectory_joint_difference')


def validate_reference(value):
    if not isinstance(value,dict) or set(value)!={'sha256','times_ns','eef'}:
        raise ValueError('固定轨迹需要完整时间表与末端目标')
    times,poses=value['times_ns'],value['eef']
    digest=value['sha256']
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid trajectory SHA256')
    if not isinstance(times,list) or not 2<=len(times)<=5000 or not isinstance(poses,list) or len(poses)!=len(times):
        raise ValueError('Invalid fixed trajectory size')
    if any(type(t) is not int or not 0<t<2**63 for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Trajectory timestamps must strictly increase')
    for p in poses:
        if (not isinstance(p,list) or len(p)!=7 or any(type(v) not in (float,int) or not math.isfinite(v) for v in p)
                or abs(sum(v*v for v in p[3:])-1)>.001):
            raise ValueError('Invalid fixed trajectory EEF target')
    return dict(sha256=digest,times_ns=list(times),eef=[list(p) for p in poses])


def validate_gains(value):
    if not isinstance(value, dict) or set(value) != set(BASELINE):
        raise ValueError('需要 Kq、Kqd、Kx、Kxd 四组参数')
    result = {}
    for key, baseline in BASELINE.items():
        row = value[key]
        if not isinstance(row, list) or len(row) != len(baseline):
            raise ValueError(f'{key} 需要 {len(baseline)} 个数值')
        if any(type(v) not in (int, float) or not math.isfinite(v)
               or not .5*b <= v <= 1.5*b for v,b in zip(row, baseline)):
            raise ValueError(f'{key} 每项须为有限数值，且在记录基准值的 0.5–1.5 倍内')
        result[key] = [float(v) for v in row]
    return result


def validate_arms(value):
    if not isinstance(value, dict) or set(value) != {'left','right'}:
        raise ValueError('需要左右臂各自的阻抗参数')
    return {arm:validate_gains(value[arm]) for arm in ('left','right')}


def schema():
    presets = []
    for name,key,factor in [('基准','Kx',1),('平移刚度 +10%','Kx',1.1),
                            ('平移刚度 +20%','Kx',1.2),('平移阻尼 −10%','Kxd',.9),
                            ('平移阻尼 +10%','Kxd',1.1)]:
        gains = {k:list(v) for k,v in BASELINE.items()}
        gains[key][:3] = [round(v*factor,6) for v in gains[key][:3]]
        presets.append(dict(name=name,gains=gains))
    return dict(baseline=BASELINE, limit_factors=LIMIT_FACTORS, formula=FORMULA, presets=presets,
                velocity_reference=VELOCITY_REFERENCE, default_velocity_mode=DEFAULT_VELOCITY_MODE,
                control_laws=[control_law(mode) for mode in ('zero', DEFAULT_VELOCITY_MODE)])
