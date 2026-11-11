"""Loopback-only, bounded access to native memory snapshots during motion.

No device calls, FK, file I/O or motion dispatch belong on this endpoint.
The command server remains the sole owner of motion and replay reservations.
"""
import pickle
import logging
import threading

import zmq


class FeedbackServer:
    def __init__(self, robot, port):
        self.robot = robot
        self.port = int(port)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='fr3-feedback-rpc', daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(2):
            self.stop_event.set()
            raise TimeoutError('read-only feedback endpoint did not start')
        if self.error:
            raise RuntimeError('read-only feedback endpoint failed: ' + str(self.error))

    def stop(self):
        self.stop_event.set()

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 2)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.MAXMSGSIZE, 65536)
        socket.setsockopt(zmq.SNDTIMEO, 50)
        try:
            socket.bind('tcp://127.0.0.1:' + str(self.port))
            self.port = int(socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(':', 1)[1])
            self.ready.set()
            while not self.stop_event.is_set():
                if not socket.poll(50, zmq.POLLIN):
                    continue
                try:
                    request = pickle.loads(socket.recv())
                    if not isinstance(request, dict):
                        raise ValueError('feedback request must be a dictionary')
                    method, args = request.get('method'), request.get('args', {})
                    if not isinstance(args, dict):
                        raise ValueError('feedback args must be a dictionary')
                    if method == 'get_control_mode' and not args:
                        # Configured mode, not an assertion that motion completed.
                        result = self.robot.control_mode
                    elif method == 'get_observations':
                        if set(args) - {'recording', 'fast'} or args.get('fast', False):
                            raise ValueError('feedback accepts native observations only')
                        result = self.robot.get_feedback_observations(recording=bool(args.get('recording', False)))
                    else:
                        raise ValueError('read-only feedback endpoint refuses method ' + str(method))
                except Exception as exc:
                    result = {'error': type(exc).__name__ + ': ' + str(exc)}
                # A stalled consumer cannot hold the serialized command server.
                socket.send(pickle.dumps(result, protocol=4))
        except Exception as exc:
            self.error = exc
            self.ready.set()
            logging.getLogger(__name__).error('Read-only feedback endpoint stopped: %s', exc)
        finally:
            socket.close(0)
            context.term()
