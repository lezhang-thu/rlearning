#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari_lz7.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

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

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 5
BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 15  # batch for efficient forward
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None

NUM_ACTIONS = None
ENV_NAME = None
NETWORK_ARCH = None  # network architecture
PSC_COLOR_MAX = 256
PSC_IMAGE_SIZE = (84, 84)
FILENAME = 'psc_data.pkl'

CLIP_PARAM = 0.1

AVG_ALPHA = 0.99


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
        reward_acc = tf.cast(reward_acc, tf.float32)
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
                l = Conv2D('conv2', l, out_channel=64, kernel_shape=3, stride=1)
        l = FullyConnected('fc0', l, 512, nl=tf.identity)

        reward_acc = FullyConnected('fc0-r', reward_acc, out_dim=128, nl=tf.nn.relu)
        reward_acc = FullyConnected('fc1-r', reward_acc, out_dim=128, nl=tf.nn.relu)
        reward_acc = FullyConnected('fc2-r', reward_acc, out_dim=128, nl=tf.identity)

        l = tf.concat([l, reward_acc], axis=1)
        l = PReLU('prelu', l)

        logits = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)  # unnormalized policy
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return logits, value

    def _build_graph(self, inputs):
        state, reward_acc, action, oldpi, vpred_old, atarg, ret = inputs

        # @lezhang.thu
        """we need old network to generate training data"""
        with tf.variable_scope('network_old'):
            logits_old, value_old = self.get_NN_prediction(state, reward_acc)

        value_old = tf.squeeze(value_old, [1], name='pred_value_old')  # (B,)
        policy_old = tf.nn.softmax(logits_old, name='policy_old')

        logits, self.value = self.get_NN_prediction(state, reward_acc)
        self.value = tf.squeeze(self.value, [1], name='pred_value_new')  # (B,)
        self.policy = tf.nn.softmax(logits, name='policy_new')

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

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt

    @staticmethod
    def update_old_param():
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            avg_name = v.op.name
            if avg_name.startswith('network_old'):
                new_name = avg_name.replace('network_old/', '')
                logger.info("{} <- {}".format(avg_name, new_name))
                ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_old_network')


class MySimulatorWorker(SimulatorProcess):
    def __init__(self, idx, pipe_c2s, pipe_s2c, joint_info, dirname, policy):
        super(MySimulatorWorker, self).__init__(idx, pipe_c2s, pipe_s2c)

        # @lezhang.thu
        self.policy = policy

        if self.policy == 'old':
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
        if self.policy == 'old':
            self.psc.psc_reward(gym_pl.current_state())
        pseudo_reward = None
        # loop invariant: S_t. Start: t=0.
        n = 0  # n is t
        while True:
            c2s_socket.send(dumps(
                # last component is is_over
                (self.identity, self.policy, state, reward, pseudo_reward, False)),
                copy=False)  # require A_t
            action = loads(s2c_socket.recv(copy=False).bytes)
            reward, is_over = player.action(action)  # get R_{t+1}
            """Bin reward to {+1, 0, -1} by its sign."""
            reward = np.sign(reward)

            if is_over:
                c2s_socket.send(dumps(
                    (self.identity, self.policy, None, reward, 0, True)),
                    copy=False)  # worker requires no action
                state = player.current_state()  # S_0
                reward = None  # for the auto-restart state
                if self.policy == 'old':
                    self.psc.psc_reward(gym_pl.current_state())  # the very first frame visited
                    pseudo_reward = None
            else:
                state = player.current_state()  # S_{t+1}
                if self.policy == 'old':
                    gym_frame = gym_pl.current_state()  # S_{t+1}'s frame
                    pseudo_reward = self.psc.psc_reward(gym_frame)

            if self.policy == 'old':
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
    class ClientState(object):
        def __init__(self):
            # (S_t, A_t, R_{t+1}, \hat{v}(S_t, w))
            self.memory = []  # list of experience
            self.prev_episode = []  # info. of the previous episode
            self.reward_acc = 0

    def __init__(self, pipe_c2s, pipe_s2c, model):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2 * LOCAL_TIME_MAX)

        self.avg_old = None
        self.avg_new = None

        from collections import defaultdict
        self.clients = defaultdict(self.ClientState)

    def _setup_graph(self):
        self._op = Model.update_old_param()
        self.async_predictor_old = MultiThreadAsyncPredictor(
            self.trainer.get_predictors(['state', 'reward_acc'], ['policy_old', 'pred_value_old'],
                                        PREDICTOR_THREAD), batch_size=PREDICT_BATCH_SIZE)
        self.async_predictor_new = MultiThreadAsyncPredictor(
            self.trainer.get_predictors(['state', 'reward_acc'], ['policy_new'],
                                        PREDICTOR_THREAD), batch_size=PREDICT_BATCH_SIZE)

    def _on_state(self, ident, state, policy):
        client = self.clients[ident]

        def cb(outputs):
            try:
                if policy == 'old':
                    distrib, value = outputs.result()  # value = \hat{v}(S_t, w)
                else:
                    distrib = outputs.result()[0]
            except CancelledError:
                logger.info("Client {} cancelled.".format(ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            # state = S_t, action = A_t, value = \hat{v}(S_t, w)
            # client's reward_acc, without acting A_t
            if policy == 'old':
                client.memory.append(TransitionExperience(
                    state, action, reward=None,
                    reward_acc=client.reward_acc, value=value, prob=distrib[action]))
            # feedback A_t, \hat{v}(S_t, w)
            self.send_queue.put([ident, dumps(action)])

        if policy == 'new':
            self.async_predictor_new.put_task([state, client.reward_acc], cb)  # state = S_t
        else:
            self.async_predictor_old.put_task([state, client.reward_acc], cb)

    def _on_episode_over(self, ident, policy):
        # @lezhang.thu
        client = self.clients[ident]

        if policy == 'new':
            if self.avg_new is None:
                self.avg_new = client.reward_acc
            else:
                self.avg_new = AVG_ALPHA * self.avg_new + (1 - AVG_ALPHA) * client.reward_acc
            return  # notice! @lezhang.thu
        else:
            if self.avg_old is None:
                self.avg_old = client.reward_acc
            else:
                self.avg_old = AVG_ALPHA * self.avg_old + (1 - AVG_ALPHA) * client.reward_acc

        client.prev_episode.clear()
        client.memory.reverse()

        """vpred_tp1 is \hat{v}(S_t+1, w)"""
        vpred_tp1 = 0.0
        gaelam = 0.0
        for idx, k in enumerate(client.memory):
            """k.value is \hat{v}(S_t, w)"""
            delta = k.reward + GAMMA * vpred_tp1 - k.value
            gaelam = delta + GAMMA * LAM * gaelam
            """tdlamret is gaelam + \hat{v}(S_t, w)"""
            client.prev_episode.append(
                [k.state, k.reward_acc, k.action, k.prob, k.value, gaelam, gaelam + k.value])
            vpred_tp1 = k.value
        """client.prev_episode[k][4] is gaelam, i.e. adv"""
        atarg = [client.prev_episode[k][4] for k in range(len(client.prev_episode))]
        atarg = np.asarray(atarg)
        """standardized advantage function estimate"""
        atarg = (atarg - atarg.mean()) / atarg.std()
        for k in range(len(client.prev_episode)):
            client.prev_episode[k][4] = atarg[k]
        client.memory.clear()  # remember!

    def _on_datapoint(self, ident, policy):
        if policy == 'new':
            return
        client = self.clients[ident]
        if len(client.prev_episode) == 0:
            return

        idxes = [
            random.randint(0, len(client.prev_episode) - 1)
            for _ in range(LOCAL_TIME_MAX)]
        for k in idxes:
            self.queue.put(client.prev_episode[k])

    def _before_train(self):
        self._op.run()
        self.async_predictor_old.start()
        self.async_predictor_new.start()

    def _trigger_step(self):
        if self.avg_new >= self.avg_old:
            self._op.run()
            self.avg_old = self.avg_new

    def _trigger_epoch(self):
        self.trainer.monitors.put_scalar('avg_new', self.avg_new)
        self.trainer.monitors.put_scalar('avg_old', self.avg_old)

    def run(self):
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False).bytes)
                ident = msg[0]
                client = self.clients[ident]

                policy, state, reward, pseudo_reward, is_over = msg[1:]  # reward is R_t, invariant (S_t, R_t)
                # TODO check history and warn about dead client

                # check if reward&is_over is valid
                # in the first message, only state is valid
                """'new' agent has no way to determine whether the first frame"""
                if reward is not None:
                    client.reward_acc += reward

                if len(client.memory) > 0:  # only policy is 'old'
                    # R_t in (S_{t-1}, A_{t-1}, R_t, \hat{v}(S_{t-1}, w)
                    client.memory[-1].reward = reward + pseudo_reward
                if is_over:
                    self._on_episode_over(ident, policy)
                    client.reward_acc = 0
                else:
                    self._on_datapoint(ident, policy)
                    # feed state and return action
                    self._on_state(ident, state, policy)

        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")


def get_shared_mem(num_proc):
    import ctypes
    from multiprocessing.sharedctypes import RawArray
    from multiprocessing import Lock

    sync_steps = STEPS_PER_EPOCH * BATCH_SIZE // num_proc
    return {
        'lock': Lock(),
        # initially zeroed
        'updated': RawArray(ctypes.c_int, num_proc),
        'sync_steps': sync_steps}


def get_config():
    dirname = os.path.join('train_log', 'train-lezhang-7-{}'.format(ENV_NAME))
    logger.set_logger_dir(dirname)
    M = Model()

    joint_info = get_shared_mem(SIMULATOR_PROC)

    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [
        MySimulatorWorker(
            k, namec2s, names2c, joint_info, dirname, policy='new')
        for k in range(SIMULATOR_PROC // 2)]
    """the starting point should be SIMULATOR_PROC // 2. NOTICE!"""
    procs.extend(
        [MySimulatorWorker(
            k, namec2s, names2c, joint_info, dirname, policy='old')
         for k in range(SIMULATOR_PROC // 2, SIMULATOR_PROC)]
    )
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
            # @lezhang.thu
            ScheduledHyperParamSetter('clip_param', [(0, CLIP_PARAM), (1000, 0.0)], interp='linear'),
            master,
            StartProcOrThread(master),
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
    args = parser.parse_args()

    ENV_NAME = args.env
    logger.info("Environment Name: {}".format(ENV_NAME))
    NUM_ACTIONS = get_player().get_action_space().num_actions()
    logger.info("Number of actions: {}".format(NUM_ACTIONS))

    NETWORK_ARCH = args.network
    logger.info("Using network architecutre: " + NETWORK_ARCH)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task != 'train':
        assert args.load is not None
        '''
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
        '''
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
