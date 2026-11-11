"""Request-local stage durations. No IO, threads, network or safety decisions."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import time
import math

current = ContextVar('mozi_node_timing', default=None)


def begin_request(identity):
    value = {'id': identity, 'start_ns': time.monotonic_ns(), 'spans': [], 'overflow': 0, 'details': {}, 'start_wall_ns': time.time_ns()}
    return current.set(value)


def end_request(token):
    value = current.get()
    result = dict(id=value['id'], node_total_ms=(time.monotonic_ns()-value['start_ns'])/1e6,
                  spans=value['spans'], overflow=value['overflow'], details=value['details'], start_monotonic_ns=value['start_ns'], start_wall_ns=value['start_wall_ns'])
    current.reset(token)
    return result


@contextmanager
def span(name):
    value = current.get()
    if value is None:
        yield
        return
    start = time.monotonic_ns()
    outcome = 'ok'
    try:
        yield
    except BaseException:
        outcome = 'error'
        raise
    finally:
        if len(value['spans']) < 64:
            value['spans'].append(dict(stage=name, start_ms=(start-value['start_ns'])/1e6,
                                       duration_ms=(time.monotonic_ns()-start)/1e6, outcome=outcome))
        else:
            value['overflow'] += 1


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)
        return call
    return decorate


def note(name, value):
    """Small bounded request-local evidence, included in the existing reply."""
    request = current.get()
    if request is None:
        return
    if len(request['details']) >= 32 and name not in request['details']:
        request['overflow'] += 1
        return
    if isinstance(value, (list, tuple)):
        value = [None if isinstance(v, float) and not math.isfinite(v) else v if isinstance(v, (str, int, float, bool, type(None))) else None for v in value[:32]]
    elif not isinstance(value, (str, int, float, bool, type(None))):
        return
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    if isinstance(value, str):
        value = value[:512]
    request['details'][name] = value
