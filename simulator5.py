#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: simulator.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Version Info.: incorporate the ideas of Its, jp

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

from tensorpack.models.common import disable_layer_logging
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
            self.memory = []  # list of Experience

            self.not_covered_index = 0
            self.not_scanned_index = 0

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
                ident, state, reward, isOver, rw_ds = msg  # reward = R_t, invariant (S_t, R_t)
                # TODO check history and warn about dead client
                client = self.clients[ident]

                # check if reward&isOver is valid
                # in the first message, only state is valid
                if len(client.memory) > 0:
                    client.memory[-1].reward = reward
                    # R_t in (S_{t-1}, A_{t-1}, R_t, \hat{v}(S_{t-1}, w)
                    client.memory[-1].rw_ds = rw_ds

                self._process_memory(ident)
                if isOver:
                    client.memory.append(
                        TransitionExperience(
                            None, None,  # state = S_t, action = A_t
                            None, value=0.0,
                            isOver=True, rw_ds=None))  # value = \hat{v}(S_t, w)
                    client.not_covered_index -= 1
                    client.not_scanned_index -= 1
                else:
                    # feed state and return action
                    self._on_state(state, ident)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    @abstractmethod
    def _on_state(self, state, ident):
        """response to state sent by ident. Preferrably an async call"""
        pass

    @abstractmethod
    def _process_memory(self, ident):
        pass

    def __del__(self):
        self.context.destroy(linger=0)



