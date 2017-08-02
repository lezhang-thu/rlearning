#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari_lz1.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# on high score learning (github "Its")

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
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient

from tensorpack.RL import *
from simulator_lz1 import *

import common
from common import (play_model, Evaluator, eval_model_multithread,
                    play_one_episode, play_n_episodes)
from pseudocount_lz0 import PSC
from tensorpack.RL.gymenv import GymEnv

if six.PY3:
    from concurrent import futures

    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 50
BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 15  # batch for efficient forward
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None
EVALUATE_PROC = min(multiprocessing.cpu_count() // 2, 20)

NUM_ACTIONS = None
ENV_NAME = None
NETWORK_ARCH = None  # Network Architecture
PSC_DOWNSAMPLE_VALUE = None
FILENAME = 'psc_data.pkl'

WINDOW_SIZE = 500  # sliding window size
DEFAULT_PROB = 1e-8
DECAY_FACTOR = 0.99
SAMPLE_STENGTH = 2
THRESHOLD = 1e-6


def get_player(viz=False, train=False, dumpdir=None, require_gym=False):
    pl = GymEnv(ENV_NAME, viz=viz, dumpdir=dumpdir)
    gym_pl = pl

    def resize(img):
        return cv2.resize(img, IMAGE_SIZE)

    def grey(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = resize(img)
        img = img[:, :, np.newaxis]
        return img

    pl = MapPlayerState(pl, grey)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    if not train:
        pl = PreventStuckPlayer(pl, 30, 1)
    else:
        pl = LimitLengthPlayer(pl, 40000)
    if require_gym: return pl, gym_pl
    return pl


class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'action_prob'),
                InputDesc(tf.float32, (None,), 'future_reward'),
                InputDesc(tf.float32, (None,), 'weight'),
                ]

    def _get_NN_prediction(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            if NETWORK_ARCH == 'tensorpack':
                l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
                l = MaxPooling('pool0', l, 2)
                l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
                l = MaxPooling('pool1', l, 2)
                l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
                l = MaxPooling('pool2', l, 2)
                l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)
            elif NETWORK_ARCH == 'nature':
                l = Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
                l = Conv2D('conv1', l, out_channel=64, kernel_shape=4, stride=2)
                l = Conv2D('conv2', l, out_channel=64, kernel_shape=3)
        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)  # unnormalized policy
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return logits, value

    def _build_graph(self, inputs):
        state, action, action_prob, future_reward, weight = inputs
        logits, self.value = self._get_NN_prediction(state)
        self.value = tf.squeeze(self.value, [1], name='pred_value')  # (B,)
        self.policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(self.policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage = tf.subtract(tf.stop_gradient(self.value), future_reward, name='advantage')

        pi_a_given_s = tf.reduce_sum(self.policy * tf.one_hot(action, NUM_ACTIONS), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance * weight, name='policy_loss')
        xentropy_loss = tf.reduce_sum(
            self.policy * log_probs, name='xentropy_loss')
        # @lezhang.thu
        # value_loss = tf.nn.l2_loss(self.value - future_reward, name='value_loss')
        value_loss = tf.reduce_sum(self.value * advantage * weight, name='value_loss')

        pred_reward = tf.reduce_mean(self.value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)

        self.cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        self.cost = tf.truediv(self.cost,
                               tf.cast(tf.shape(future_reward)[0], tf.float32),
                               name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage,
                                   self.cost, tf.reduce_mean(importance, name='importance'))

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 0.001, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorWorker(SimulatorProcess):
    def __init__(self, idx, pipe_c2s, pipe_s2c, joint_info, dirname):
        super(MySimulatorWorker, self).__init__(idx, pipe_c2s, pipe_s2c)

        self.psc = PSC(PSC_DOWNSAMPLE_VALUE)

        self.lock = joint_info['lock']
        self.updated = joint_info['updated']
        self.sync_steps = joint_info['sync_steps']
        self.file_path = os.path.join(dirname, FILENAME)

        if os.path.isfile(self.file_path):
            self.read_joint()
        self._init_window()

    def _init_window(self):
        self.w = {
            'probs': np.array([DEFAULT_PROB for _ in range(WINDOW_SIZE)],
                              dtype=np.float32),
            'valid': [False for _ in range(WINDOW_SIZE)],
            'index': 0,
            'value': np.zeros(WINDOW_SIZE, dtype=np.float32),
            'reward': np.zeros(WINDOW_SIZE, dtype=np.float32)
        }

    def _get_sample(self, bootstrap):
        sample_probs = self.w['probs'] / np.sum(self.w['probs'])
        sample_idx = np.random.choice(len(self.w['probs']), p=sample_probs)
        if not self.w['valid'][sample_idx]:
            return -1, None, None
        else:
            future_reward = self._get_return(sample_idx, bootstrap)
            return sample_idx, \
                   np.clip(1.0 / (sample_probs[sample_idx] * WINDOW_SIZE), 0, 1), \
                   future_reward

    def _get_return(self, sample_idx, bootstrap):
        # invariant: v corresponds to k's bootstrap return
        k = self.w['index']
        v = np.clip(self.w['reward'][k], -1, 1) + GAMMA * bootstrap
        while k != sample_idx:
            k = k - 1 if k > 0 else WINDOW_SIZE - 1
            v = np.clip(self.w['reward'][k], -1, 1) + GAMMA * v
        return v

    def _update_window(self, reward, value, reward_found):
        self.w['index'] = (self.w['index'] + 1) % WINDOW_SIZE
        k = self.w['index']
        self.w['valid'][k] = True
        self.w['value'][k] = value  # \hat{v}(S_t, w)
        self.w['reward'][k] = reward  # R_t+1
        self.w['probs'][k] = DEFAULT_PROB

        if reward_found:
            # update probs
            decay_prob = 1.0
            n = 0
            # invariant: n is the number of slots visited
            while n < WINDOW_SIZE and self.w['valid'][k]:
                self.w['probs'][k] += decay_prob

                k = k - 1 if k > 0 else WINDOW_SIZE - 1
                decay_prob *= DECAY_FACTOR
                n += 1

    def _episode_over_sample(self, c2s_socket):
        k = (self.w['index'] + 1) % WINDOW_SIZE

        # last just in window, it needs WINDOW_SIZE samplings
        for _ in range(WINDOW_SIZE):
            idx, weight, future_reward = self._get_sample(0.0)
            if idx != -1:
                c2s_socket.send(dumps(
                    (self.identity, 'feed', idx, weight, future_reward)),
                    copy=False)  # feed a sampled transition
            self.w['valid'][k] = False  # the window is sliding away
            self.w['probs'][k] = DEFAULT_PROB
            k = (k + 1) % WINDOW_SIZE

    def _feed_transition(self, value, c2s_socket):
        idx, weight, future_reward = self._get_sample(value)
        if idx != -1:
            c2s_socket.send(dumps(
                (self.identity, 'feed', idx, weight, future_reward)),
                copy=False)  # feed a sampled transition

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
        # loop invariant: S_t. Start: t=0.
        n = 0  # n is t
        while True:
            c2s_socket.send(dumps(
                # last component is is_over
                (self.identity, 'request', state, reward, False)),
                copy=False)  # require A_t
            action, value = loads(s2c_socket.recv(copy=False).bytes)  # A_t, \hat{v}(S_t, w)
            self._feed_transition(value, c2s_socket)  # as we have \hat{v}(S_t, w)

            reward, is_over = player.action(action)  # get R_{t+1}
            if is_over:
                c2s_socket.send(dumps(
                    (self.identity, 'request', None, reward, True)),
                    copy=False)  # worker requires no action
                self._update_window(reward, value, not abs(reward - 0.0) < THRESHOLD)
                self._episode_over_sample(c2s_socket)
                state = player.current_state()  # S_0
                reward = None  # for the auto-restart state
            else:
                """assume S_t-1 etc. is okay, i.e. in the window,
                here, S_t gets its info.:
                R_t+1, \hat{v}(S_t, w).
                this is the invariant."""
                reward_found = not abs(reward - 0.0) < THRESHOLD
                state = player.current_state()  # S_{t+1}
                gym_frame = gym_pl.current_state()  # S_{t+1}'s frame
                reward += self.psc.psc_reward(gym_frame)
                self._update_window(reward, value, reward_found)

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


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2 * (SAMPLE_STENGTH + 1))

        from collections import defaultdict
        self.windows = defaultdict(lambda:
                                   {'window': [None for _ in range(WINDOW_SIZE)],
                                    'buffer': None,
                                    'index': 0})

    def _setup_graph(self):
        self.async_predictor = MultiThreadAsyncPredictor(
            self.trainer.get_predictors(['state'], ['policy', 'pred_value'],
                                        PREDICTOR_THREAD), batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _slide_window(self, ident, transition):
        w = self.windows[ident]
        if w['buffer'] is not None:
            w['index'] = (w['index'] + 1) % WINDOW_SIZE
            w['window'][w['index']] = w['buffer']
        w['buffer'] = transition

    def _window_sample(self, ident, idx, weight, future_reward):
        w = self.windows[ident]
        for _ in range(SAMPLE_STENGTH):
            self.queue.put(w['window'][idx] + [future_reward, weight])

    def _on_state(self, state, ident):
        def cb(outputs):
            try:
                distrib, value = outputs.result()  # value = \hat{v}(S_t, w)
            except CancelledError:
                logger.info("Client {} cancelled.".format(ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client = self.clients[ident]
            # state = S_t, action = A_t, value = \hat{v}(S_t, w)
            client.memory.append(TransitionExperience(
                state, action, reward=None, value=value, prob=distrib[action]))
            self._slide_window(ident, [state, action, distrib[action]])
            # feedback A_t, \hat{v}(S_t, w)
            self.send_queue.put([ident, dumps((action, value))])

        self.async_predictor.put_task([state], cb)  # state = S_t

    def _on_episode_over(self, ident):
        self._parse_memory(0.0, ident, True)

    def _on_datapoint(self, ident):
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            bootstrap = client.memory[-1].value
            self._parse_memory(bootstrap, ident, False)

    def _parse_memory(self, init_r, ident, is_over):
        client = self.clients[ident]
        mem = client.memory
        if not is_over:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        future_reward = float(init_r)
        for idx, k in enumerate(mem):
            future_reward = np.clip(k.reward, -1, 1) + GAMMA * future_reward
            self.queue.put([k.state, k.action, k.prob, future_reward, 1.0])
        client.memory = [] if is_over else [last]


def get_shared_mem(num_proc):
    import ctypes
    from multiprocessing.sharedctypes import RawArray
    from multiprocessing import Lock

    # sync_steps = 20 * 30000  # @tensorflow-rl 20*q_target_update_steps
    sync_steps = STEPS_PER_EPOCH * BATCH_SIZE // num_proc
    return {
        'lock': Lock(),
        # initially zeroed
        'updated': RawArray(ctypes.c_int, num_proc),
        'sync_steps': sync_steps}


def get_config():
    dirname = os.path.join('train_log', 'train-lezhang-1-{}'.format(ENV_NAME))
    logger.set_logger_dir(dirname)
    M = Model()

    joint_info = get_shared_mem(SIMULATOR_PROC)

    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c, joint_info, dirname) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, M)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    return TrainConfig(
        model=M,
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            master,
            StartProcOrThread(master),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                every_k_epochs=3),
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

    parser.add_argument('--network', help='network architecture', choices=['nature', 'tensorpack'],
                        default='nature')
    parser.add_argument('--PSC_DOWNSAMPLE_VALUE', help='pseudo-count downsample max value',
                        default='128')
    args = parser.parse_args()

    ENV_NAME = args.env
    logger.info("Environment Name: {}".format(ENV_NAME))
    NUM_ACTIONS = get_player().get_action_space().num_actions()
    logger.info("Number of actions: {}".format(NUM_ACTIONS))

    PSC_DOWNSAMPLE_VALUE = int(args.PSC_DOWNSAMPLE_VALUE)
    NETWORK_ARCH = args.network
    logger.info("Using network architecutre: " + NETWORK_ARCH)

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
