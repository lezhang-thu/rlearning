from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from multiprocessing import RawArray, Barrier, Lock
import tensorflow as tf
import logging_utils
import time
from shared_utils import SharedCounter, SharedVars
import ctypes
import argparse
from policy_based_actor_learner import *

logger = logging_utils.getLogger('main')


def main(args):
    logger.debug('Config: {}'.format(args))

    """ Set up the graph, the agents, and run the agents in parallel. """
    import atari_environment
    num_actions, _, _ = atari_environment.get_actions(args.game)

    args.summ_base_dir = 'train_data/sum_log/{}/{}'.format(args.game, time.time())

    args.learning_vars = SharedVars(num_actions)
    if args.opt_mode == 'shared':
        args.opt_state = SharedVars(num_actions, opt_type=args.opt_type, lr=args.initial_lr)
    else:
        args.opt_state = None

    args.cts_updated = RawArray(ctypes.c_int, args.num_actor_learners)
    args.cts_lock = Lock()
    args.cts_sync_steps = 20 * 30000  # @tensorflow-rl 20*q_target_update_steps

    args.barrier = Barrier(args.num_actor_learners)
    args.global_step = SharedCounter(0)
    args.num_actions = num_actions

    if (args.visualize == 2): args.visualize = 0
    actor_learners = []
    for i in range(args.num_actor_learners):
        if (args.visualize == 2) and (i == args.num_actor_learners - 1):
            args.args.visualize = 1

        args.actor_id = i

        actor_learners.append(A3CLearner(args))
        actor_learners[-1].start()

    for t in actor_learners:
        t.join()

    logger.debug('All training threads finished')
    logger.debug('All threads stopped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', help='Name of game')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-v', '--visualize', default=0, type=int,
                        help='0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized. Default = 0',
                        dest='visualize')

    parser.add_argument('--opt_type', default='rmsprop',
                        help='Type of optimizer: rmsprop, momentum, adam. Default = rmsprop', dest='opt_type')
    parser.add_argument('--opt_mode', default='shared',
                        help='Whether to use \'local\' or \'shared\' vector(s) for the moving average(s). Default = shared',
                        dest='opt_mode')

    # consistent with tensorflow beta1, beta2
    parser.add_argument('--b1', default=0.9, type=float, help='beta1 for the Adam optimizer. Default = 0.9', dest='b1')
    parser.add_argument('--b2', default=0.999, type=float, help='beta2 for the Adam optimizer. Default = 0.999',
                        dest='b2')
    # TODO @tensorpack AdamOptimizer's epsilon = 1e-3. tensorflow default is 1e-08
    # TODO tensorflow default for RMSPropOptimizer is epsilon=1e-10
    parser.add_argument('--e', default=0.001, type=float,
                        help='Epsilon for the Rmsprop and Adam optimizers. Default = 1e-3', dest='e')
    # TODO tensorflow default for RMSPropOptimizer is decay=0.9
    parser.add_argument('--alpha', default=0.9, type=float,
                        help='Discounting factor for the history/coming gradient, for the Rmsprop optimizer. Default = 0.9',
                        dest='alpha')

    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int,
                        help='Number of global steps during which the learning rate will be linearly annealed towards zero. Default = 80*10^6',
                        dest='lr_annealing_steps')
    parser.add_argument('-n', '--num_actor_learners', default=16, type=int,
                        help='number of actors (processes). Default = 16', dest='num_actor_learners')
    parser.add_argument('--max_global_steps', default=80000000, type=int,
                        help='Max number of training steps. Default = 80*10^6', dest='max_global_steps')

    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.initial_lr = 0.001  # tensorpack
    args.max_local_steps = 32  # @lezhang batch_size

    args.opt_type = 'rmsprop'
    args.opt_mode = 'shared'
    args.e = 1e-3

    args.alg_type = 'a3c'
    args.gamma = 0.99
    main(args)
