"""Optional, bounded recording side channel. Never owns hardware or sends commands.

Producers only enqueue immutable snapshots. JSON and ZeroMQ run in the worker.
Sequence/session IDs and idle heartbeats make transport/queue losses observable.
"""
import json
import os
import queue
import threading
import time
import uuid


class Publisher:
    def __init__(self, endpoint, stream, capacity=2048):
        self.endpoint, self.stream = endpoint, stream
        self.session = uuid.uuid4().hex
        self.queue = queue.Queue(maxsize=capacity)
        self.sequence = self.dropped = self.sent_sequence = 0
        self.last_source_ns = 0
        self.error = ''
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='record-telemetry')
        self.thread.start()

    def emit(self, data, source_ns):
        # One producer per publisher. No socket, file, RPC or serialization here.
        self.sequence += 1
        self.last_source_ns = int(source_ns)
        try:
            self.queue.put_nowait((self.sequence, int(source_ns), data))
        except queue.Full:
            self.dropped += 1

    def _send(self, socket, value):
        import zmq
        socket.send(json.dumps(value, allow_nan=False, separators=(',', ':')).encode(), zmq.NOBLOCK)

    def _run(self):
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 2048)
        try:
            socket.bind(self.endpoint)
            heartbeat = 0.
            while not self.stop.is_set():
                if time.monotonic() - heartbeat >= 1:
                    self._send(socket, dict(version=1, kind='status', stream=self.stream,
                        session=self.session, source_ns=time.time_ns(), sequence=self.sequence,
                        sent_sequence=self.sent_sequence, dropped_total=self.dropped,
                        last_source_ns=self.last_source_ns, error=self.error))
                    heartbeat = time.monotonic()
                try:
                    sequence, stamp, data = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                try:
                    self._send(socket, dict(version=1, kind='sample', stream=self.stream,
                        session=self.session, sequence=sequence, source_ns=stamp,
                        dropped_total=self.dropped, data=data))
                    self.sent_sequence = sequence
                except (ValueError, TypeError, zmq.ZMQError) as exc:
                    self.error = str(exc)[:160]
                    self.dropped += 1
        except Exception as exc:
            self.error = str(exc)[:160]
        finally:
            socket.close()
            context.term()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.3)


def configured(stream):
    endpoint = os.environ.get('MOZI_RECORDING_TELEMETRY_ENDPOINT', '')
    return Publisher(endpoint, stream) if endpoint else None


def emit_ik(publisher, joints, position, quaternion, mode, submitted_ns, accepted_ns, tracking_velocity=None):
    if publisher is not None:
        data=dict(joint_positions=tuple(float(x) for x in joints),
            eef=tuple(float(x) for x in position) + tuple(float(x) for x in quaternion),
            accepted=True, ik_mode=mode, submitted_ns=int(submitted_ns),
            semantics='post_ik_float32_target_accepted_by_polymetis')
        if tracking_velocity is not None:
            data['tracking_velocity']=tracking_velocity
        publisher.emit(data, accepted_ns)


def emit_gripper(publisher, width, stamp, max_width):
    if publisher is not None:
        import math
        valid = math.isfinite(width) and 0 <= width <= max_width
        publisher.emit(dict(width_m=float(width) if valid else None,
            raw_width_m=float(width) if math.isfinite(width) else None,
            valid=valid, error_code=0 if valid else 4,
            source='das_encoder', timestamp_source='serial_callback_host_wall_ns',
            force=None, enabled=None), stamp)
