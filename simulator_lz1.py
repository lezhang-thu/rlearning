#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: simulator_lz1.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import multiprocessing as mp
import time
import os
import threading
from abc import abstractmethod, ABCMeta
from collections import defaultdict

import six
from six.moves import queue
import zmq

from tensorpack.callbacks import Callback
from tensorpack.tfutils.varmanip import SessionUpdate
from tensorpack.predict import OfflinePredictor
from tensorpack.utils import logger
from tensorpack.utils.serialize import loads, dumps
from tensorpack.utils.concurrency import LoopThread, ensure_proc_terminate

__all__ = ['SimulatorProcess', 'SimulatorMaster',
           'SimulatorProcessStateExchange',
           'TransitionExperience']


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


@six.add_metaclass(ABCMeta)
class SimulatorProcessBase(mp.Process):
    def __init__(self, idx):
        super(SimulatorProcessBase, self).__init__()
        self.idx = int(idx)
        self.name = u'simulator-{}'.format(self.idx)
        self.identity = self.name.encode('utf-8')

    @abstractmethod
    def _build_player(self):
        pass


class SimulatorProcessStateExchange(SimulatorProcessBase):
    """
    A process that simulates a player and communicates to master to
    send states and receive the next action
    """

    def __init__(self, idx, pipe_c2s, pipe_s2c):
        """
        :param idx: idx of this process
        """
        super(SimulatorProcessStateExchange, self).__init__(idx)
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

    def run(self):
        pass


# compatibility
SimulatorProcess = SimulatorProcessStateExchange


class SimulatorMaster(threading.Thread):
    """ A base thread to communicate with all StateExchangeSimulatorProcess.
        It should produce action for each simulator, as well as
        defining callbacks when a transition or an episode is finished.
    """

    class ClientState(object):
        def __init__(self):
            # (S_t, A_t, R_{t+1}, \hat{v}(S_t, w))
            self.memory = []  # list of experience

    def __init__(self, pipe_c2s, pipe_s2c):
        super(SimulatorMaster, self).__init__()
        assert os.name != 'nt', "Doesn't support windows!"
        self.daemon = True
        self.name = 'SimulatorMaster'

        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.c2s_socket.set_hwm(10)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.s2c_socket.set_hwm(10)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=100)

        def f():
            msg = self.send_queue.get()
            self.s2c_socket.send_multipart(msg, copy=False)

        self.send_thread = LoopThread(f)
        self.send_thread.daemon = True
        self.send_thread.start()

        self.clients = defaultdict(self.ClientState)

        # make sure socket get closed at the end
        def clean_context(soks, context):
            for s in soks:
                s.close()
            context.term()

        import atexit
        atexit.register(clean_context, [self.c2s_socket, self.s2c_socket], self.context)

    def run(self):
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False).bytes)
                ident = msg[0]

                if msg[1] == 'request':  # normal requiring A_t
                    state, reward, is_over = msg[2:]  # reward is R_t, invariant (S_t, R_t)
                    # TODO check history and warn about dead client
                    client = self.clients[ident]

                    # check if reward&is_over is valid
                    # in the first message, only state is valid
                    if len(client.memory) > 0:
                        # R_t in (S_{t-1}, A_{t-1}, R_t, \hat{v}(S_{t-1}, w)
                        client.memory[-1].reward = reward
                    if is_over:
                        # before get any 'feed' msg, the master already clear the buffer
                        self._slide_window(ident, None)
                        self._on_episode_over(ident)
                    else:
                        self._on_datapoint(ident)
                        # feed state and return action
                        self._on_state(state, ident)
                elif msg[1] == 'feed':  # sliding window sampling
                    idx, weight, future_reward = msg[2:]
                    self._window_sample(ident, idx, weight, future_reward)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    @abstractmethod
    def _on_state(self, state, ident):
        """response to state sent by ident. Preferrably an async call"""

    @abstractmethod
    def _on_episode_over(self, client):
        """ callback when the client just finished an episode.
            You may want to clear the client's memory in this callback.
        """

    def _on_datapoint(self, client):
        """ callback when the client just finished a transition"""

    @abstractmethod
    def _window_sample(self, ident, idx, hat_v):
        """add a sample for the sliding window to queue"""

    @abstractmethod
    def _slide_window(self, ident, transition):
        """feed a new transition to sliding window system"""

    def __del__(self):
        self.context.destroy(linger=0)


if __name__ == '__main__':
    import random
    from tensorpack.RL import NaiveRLEnvironment


    class NaiveSimulator(SimulatorProcess):
        def _build_player(self):
            return NaiveRLEnvironment()


    class NaiveActioner(SimulatorMaster):
        def _get_action(self, state):
            time.sleep(1)
            return random.randint(1, 12)

        def _on_episode_over(self, client):
            # print("Over: ", client.memory)
            client.memory = []
            client.state = 0


    name = 'ipc://whatever'
    procs = [NaiveSimulator(k, name) for k in range(10)]
    [k.start() for k in procs]

    th = NaiveActioner(name)
    ensure_proc_terminate(procs)
    th.start()

    time.sleep(100)
