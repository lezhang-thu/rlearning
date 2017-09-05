#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari_lz8.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
""" version: Build experience replay with proximal
policy optimization
"""

import numpy as np
import os
import sys
import time
import random
import uuid
import argparse
import multiprocessing
import threading
import pickle

import cv2
import tensorflow as tf
import six
from six.moves import queue
import zmq

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.serialize import *
from tensorpack.utils.stats import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient

from tensorpack.RL import *
from simulator_lz6 import *

import common
from common import (play_model, Evaluator, eval_model_multithread,
                    play_one_episode, play_n_episodes)
from pseudocount_lz6 import PSC
from tensorpack.RL.gymenv import GymEnv

if six.PY3:
    from concurrent import futures

    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
LAM = 0.95
CHANNEL = FRAME_HISTORY
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 5
BATCH_SIZE = 64 
PREDICT_BATCH_SIZE = 4  # batch for efficient forward
"""expreplay longer w.r.t. time, hence less agents"""
SIMULATOR_PROC = 8

PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None

NUM_ACTIONS = None
ENV_NAME = None
PSC_COLOR_MAX = 256
"""down-sample is needed; otherwise too small prob."""
PSC_IMAGE_SIZE = (42, 42)
FILENAME = 'psc_data.pkl'
SYNC_STEPS = 1e5

CLIP_PARAM = 0.2
MEMORY_SIZE = 2e4
UPDATE_FREQ = BATCH_SIZE


def get_player(viz=False, train=False, dumpdir=None, require_gym=False):
    pl = GymEnv(ENV_NAME, viz=viz, dumpdir=dumpdir)
    gym_pl = pl

    def resize(img):
        return cv2.resize(img, IMAGE_SIZE)

    def gray(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = resize(img)
        img = img[:, :, np.newaxis]
        return img.astype(np.uint8)  # to save some memory

    pl = MapPlayerState(pl, gray)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    if not train:
        pl = PreventStuckPlayer(pl, 30, 1)
    else:
        pl = LimitLengthPlayer(pl, 60000)
    if require_gym: return pl, gym_pl
    return pl


class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [
            InputDesc(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
            InputDesc(tf.int32, (None,), 'reward_acc'),
            InputDesc(tf.int64, (None,), 'action'),
            InputDesc(tf.float32, (None,), 'action_prob'),
            InputDesc(tf.float32, (None,), 'value'),
            InputDesc(tf.float32, (None,), 'gaelam'),
            InputDesc(tf.float32, (None,), 'tdlamret'),
        ]

    # decorate the function
    @auto_reuse_variable_scope
    def get_NN_prediction(self, image, reward_acc):
        return self._get_NN_prediction(image, reward_acc)

    def _get_NN_prediction(self, image, reward_acc):
        image = tf.cast(image, tf.float32) / 255.0
        # reward_acc = tf.cast(reward_acc, tf.float32)
        with argscope(Conv2D, nl=PReLU.symbolic_function, use_bias=True), \
             argscope(LeakyReLU, alpha=0.01):
            l = (LinearWrap(image)
                 # Nature architecture
                 .Conv2D('conv0', out_channel=32, kernel_shape=8, stride=4)
                 .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                 .Conv2D('conv2', out_channel=64, kernel_shape=3)

                 # architecture used for the figure in the README, slower but takes fewer iterations to converge
                 # .Conv2D('conv0', out_channel=32, kernel_shape=5)
                 # .MaxPooling('pool0', 2)
                 # .Conv2D('conv1', out_channel=32, kernel_shape=5)
                 # .MaxPooling('pool1', 2)
                 # .Conv2D('conv2', out_channel=64, kernel_shape=4)
                 # .MaxPooling('pool2', 2)
                 # .Conv2D('conv3', out_channel=64, kernel_shape=3)

                 .FullyConnected('fc0', 512, nl=LeakyReLU)())

        # reward_acc = FullyConnected('fc0-r', reward_acc, out_dim=128, nl=tf.nn.relu)
        # reward_acc = FullyConnected('fc1-r', reward_acc, out_dim=128, nl=tf.nn.relu)
        # reward_acc = FullyConnected('fc2-r', reward_acc, out_dim=128, nl=tf.identity)

        # l = tf.concat([l, reward_acc], axis=1)

        logits = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)  # unnormalized policy
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return logits, value

    def _build_graph(self, inputs):
        state, reward_acc, action, oldpi, vpred_old, atarg, ret = inputs

        logits, self.value = self._get_NN_prediction(state, reward_acc)
        self.value = tf.squeeze(self.value, [1], name='pred_value')  # (B,)
        self.policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(self.policy + 1e-6)

        pi = tf.reduce_sum(self.policy * tf.one_hot(action, NUM_ACTIONS), 1)  # (B,)
        ratio = pi / (oldpi + 1e-8)  # pnew / pold
        clip_param = tf.get_variable(
            'clip_param', shape=[],
            initializer=tf.constant_initializer(CLIP_PARAM), trainable=False)
        surr1 = ratio * atarg  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
        """PPO's pessimistic surrogate (L^CLIP)"""
        pol_surr = - tf.reduce_sum(tf.minimum(surr1, surr2))

        vfloss1 = tf.square(self.value - ret)
        vpredclipped = vpred_old + \
                       tf.clip_by_value(self.value - vpred_old, -clip_param, clip_param)
        vfloss2 = tf.square(vpredclipped - ret)
        """we do the same clipping-based trust region for the value function"""
        vf_loss = .5 * tf.reduce_sum(tf.maximum(vfloss1, vfloss2))

        xentropy_loss = tf.reduce_sum(
            self.policy * log_probs, name='xentropy_loss')

        pred_reward = tf.reduce_mean(self.value, name='predict_reward')
        advantage = symbf.rms(atarg, name='rms_advantage')
        entropy_beta = tf.get_variable(
            'entropy_beta', shape=[],
            initializer=tf.constant_initializer(0.01), trainable=False)

        self.cost = tf.add_n([pol_surr, xentropy_loss * entropy_beta, vf_loss])
        self.cost = tf.truediv(self.cost,
                               tf.cast(tf.shape(action)[0], tf.float32),
                               name='cost')
        summary.add_moving_summary(pol_surr, xentropy_loss,
                                   vf_loss, pred_reward, advantage,
                                   self.cost)

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 0.001, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        # gradprocs = [SummaryGradient()]
        # @lezhang.thu

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]

        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorWorker(SimulatorProcess):
    def __init__(self, idx, pipe_c2s, pipe_s2c, joint_info, dirname):
        super(MySimulatorWorker, self).__init__(idx, pipe_c2s, pipe_s2c)

        self.psc = PSC(PSC_IMAGE_SIZE, PSC_COLOR_MAX)

        self.lock = joint_info['lock']
        self.updated = joint_info['updated']
        self.sync_steps = joint_info['sync_steps']
        self.file_path = os.path.join(dirname, FILENAME)

        if os.path.isfile(self.file_path):
            self._read_joint()

    def run(self):
        player, gym_pl = self._build_player()

        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(2)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        # s2c_socket.set_hwm(5)
        s2c_socket.connect(self.s2c)

        state = player.current_state()  # S_0
        reward = None  # R_0 serves as dummy
        self.psc.psc_reward(gym_pl.current_state())
        pseudo_reward = None
        # loop invariant: S_t. Start: t=0.
        n = 0  # n is t
        while True:
            c2s_socket.send(dumps(
                # last component is is_over
                (self.identity, state, reward, pseudo_reward, False)),
                copy=False)  # require A_t
            action = loads(s2c_socket.recv(copy=False).bytes)
            reward, is_over = player.action(action)  # get R_{t+1}
            """Bin reward to {+1, 0, -1} by its sign."""
            reward = np.sign(reward)

            if is_over:
                c2s_socket.send(dumps(
                    (self.identity, None, reward, 0, True)),
                    copy=False)  # worker requires no action
                state = player.current_state()  # S_0
                reward = None  # for the auto-restart state
                self.psc.psc_reward(gym_pl.current_state())  # the very first frame visited
                pseudo_reward = None
            else:
                state = player.current_state()  # S_{t+1}
                gym_frame = gym_pl.current_state()  # S_{t+1}'s frame
                pseudo_reward = self.psc.psc_reward(gym_frame)

            n += 1
            self._update_joint(n)

    def _update_joint(self, n):
        if n % self.sync_steps == 0:
            self._write_joint()
            self._read_joint()

    def _write_joint(self):
        with self.lock:
            if self.updated[self.idx] == 1:
                return
            raw_data = pickle.dumps(self.psc.get_state())
            with open(self.file_path, 'wb') as f:
                f.write(raw_data)
            for i in range(len(self.updated)):
                self.updated[i] = 1

    def _read_joint(self):
        with open(self.file_path, 'rb') as f:
            raw_data = f.read()
        self.psc.set_state(pickle.loads(raw_data))
        self.updated[self.idx] = 0

    def _build_player(self):
        return get_player(train=True, require_gym=True)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """state, reward_acc, action, oldpi, vpred_old, atarg, ret"""
        state_s, reward_acc_s, action_s, oldpi_s, vpred_old_s, atarg_s, ret_s = \
            [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, reward_acc, action, oldpi, vpred_old, atarg, ret = data
            state_s.append(np.array(state, copy=False))
            reward_acc_s.append(reward_acc)
            action_s.append(action)
            oldpi_s.append(oldpi)
            vpred_old_s.append(vpred_old)
            atarg_s.append(atarg)
            ret_s.append(ret)

        return np.array(state_s), np.array(reward_acc_s), \
               np.array(action_s), np.array(oldpi_s), np.array(vpred_old_s), \
               np.array(atarg_s), np.array(ret_s)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class MySimulatorMaster(SimulatorMaster, Callback, DataFlow):
    class ClientState(object):
        def __init__(self):
            # (S_t, A_t, R_{t+1}, \hat{v}(S_t, w))
            self.memory = []  # list of experience
            self.reward_acc = 0

    def __init__(self, pipe_c2s, pipe_s2c, model):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model

        from collections import defaultdict
        self.clients = defaultdict(self.ClientState)

        self.expreplay = ReplayBuffer(int(MEMORY_SIZE))
        self.coins = threading.Semaphore(0)

    def _setup_graph(self):
        self.async_predictor = MultiThreadAsyncPredictor(
            self.trainer.get_predictors(['state', 'reward_acc'], ['policy', 'pred_value'],
                                        PREDICTOR_THREAD), batch_size=PREDICT_BATCH_SIZE)

    def get_data(self):
        while True:
            self.coins.acquire()
            yield self.expreplay.sample(BATCH_SIZE)

    def _on_state(self, ident, state):
        client = self.clients[ident]

        def cb(outputs):
            try:
                distrib, value = outputs.result()  # value = \hat{v}(S_t, w)
            except CancelledError:
                logger.info("Client {} cancelled.".format(ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            # state = S_t, action = A_t, value = \hat{v}(S_t, w)
            # client's reward_acc, without acting A_t
            client.memory.append(TransitionExperience(
                state, action, reward=None,
                reward_acc=client.reward_acc, value=value, prob=distrib[action]))
            # feedback A_t, \hat{v}(S_t, w)
            self.send_queue.put([ident, dumps(action)])

        self.async_predictor.put_task([state, client.reward_acc], cb)

    def _on_episode_over(self, ident):
        # @lezhang.thu
        client = self.clients[ident]
        t_list = []
        """vpred_tp1 is \hat{v}(S_t+1, w)"""
        vpred_tp1 = 0.0
        gaelam = 0.0
        for idx, k in enumerate(client.memory):
            """k.value is \hat{v}(S_t, w)"""
            delta = k.reward + GAMMA * vpred_tp1 - k.value
            gaelam = delta + GAMMA * LAM * gaelam
            """tdlamret is gaelam + \hat{v}(S_t, w)"""
            t_list.append(
                [k.state, k.reward_acc, k.action, k.prob, k.value, gaelam, gaelam + k.value])
            vpred_tp1 = k.value
        """t_list[k][4] is gaelam, i.e. adv"""
        atarg = [t_list[k][4] for k in range(len(t_list))]
        atarg = np.asarray(atarg)
        """standardized advantage function estimate"""
        atarg = (atarg - atarg.mean()) / atarg.std()
        for k in range(len(t_list)):
            t_list[k][4] = atarg[k]

        for e in t_list:
            self.expreplay.add(e)
        for _ in range(len(t_list) // UPDATE_FREQ):
            self.coins.release()

    def _before_train(self):
        self.async_predictor.start()

    def run(self):
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False).bytes)
                ident = msg[0]
                client = self.clients[ident]

                state, reward, pseudo_reward, is_over = msg[1:]  # reward is R_t, invariant (S_t, R_t)
                # TODO check history and warn about dead client

                # check if reward&is_over is valid
                # in the first message, only state is valid
                if reward is not None:
                    client.reward_acc += reward

                if len(client.memory) > 0:
                    # R_t in (S_{t-1}, A_{t-1}, R_t, \hat{v}(S_{t-1}, w)
                    # @lezhang.thu, handcrafted, pseudo_reward or not
                    # client.memory[-1].reward = reward + pseudo_reward
                    client.memory[-1].reward = reward
                if is_over:
                    self._on_episode_over(ident)
                    client.reward_acc = 0
                    client.memory.clear()  # remember!
                else:
                    # feed state and return action
                    self._on_state(ident, state)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")


def get_shared_mem(num_proc):
    import ctypes
    from multiprocessing.sharedctypes import RawArray
    from multiprocessing import Lock

    return {
        'lock': Lock(),
        # initially zeroed
        'updated': RawArray(ctypes.c_int, num_proc),
        'sync_steps': int(SYNC_STEPS)}


def get_config():
    dirname = os.path.join('train_log', 'train-lezhang-8-expreplay-{}'.format(ENV_NAME))
    logger.set_logger_dir(dirname)
    M = Model()

    joint_info = get_shared_mem(SIMULATOR_PROC)

    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c, joint_info, dirname)
             for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, M)
    return TrainConfig(
        model=M,
        dataflow=master,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            # @lezhang.thu
            ScheduledHyperParamSetter('clip_param', [(0, CLIP_PARAM), (1000, 0.0)], interp='linear'),
            master,
            StartProcOrThread(master),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                every_k_epochs=1),
        ],
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--env', help='env', required=True)
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train', 'gen_submit'], default='train')
    parser.add_argument('--output', help='output directory for submission', default='output_dir')
    parser.add_argument('--episode', help='number of episode to eval',
                        default=100, type=int)
    args = parser.parse_args()

    ENV_NAME = args.env
    logger.info("Environment Name: {}".format(ENV_NAME))
    NUM_ACTIONS = get_player().get_action_space().num_actions()
    logger.info("Number of actions: {}".format(NUM_ACTIONS))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task != 'train':
        assert args.load is not None
        cfg = PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['policy'])
        if args.task == 'play':
            play_model(cfg, get_player(viz=0.01))
        elif args.task == 'eval':
            eval_model_multithread(cfg, args.episode, get_player)
        elif args.task == 'gen_submit':
            play_n_episodes(
                get_player(train=False, dumpdir=args.output),
                OfflinePredictor(cfg), args.episode)
            # gym.upload(output, api_key='xxx')
    else:
        nr_gpu = get_nr_gpu()
        if nr_gpu > 0:
            if nr_gpu > 1:
                predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
            else:
                predict_tower = [0]
            PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
            train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
            logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
                ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
            trainer = AsyncMultiGPUTrainer
        else:
            logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
            nr_gpu = 0
            PREDICTOR_THREAD = 1
            predict_tower, train_tower = [0], [0]
            trainer = QueueInputTrainer
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        config.tower = train_tower
        config.predict_tower = predict_tower
        trainer(config).train()
