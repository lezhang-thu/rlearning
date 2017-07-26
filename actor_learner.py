import numpy as np
from multiprocessing import Process
import multiprocessing
import logging_utils
import tensorflow as tf
import ctypes
import pyximport

pyximport.install()
from hogupdatemv import copy, apply_grads_mom_rmsprop, apply_grads_adam
import time
import utils
from contextlib import contextmanager

CHECKPOINT_INTERVAL = 500000

logger = logging_utils.getLogger('actor_learner')


class ActorLearner(Process):
    def __init__(self, args):
        super(ActorLearner, self).__init__()

        self.summ_base_dir = args.summ_base_dir

        self.local_step = 0
        self.global_step = args.global_step

        self.saver = None
        self.actor_id = args.actor_id
        self.alg_type = args.alg_type
        self.max_local_steps = args.max_local_steps
        self.optimizer_type = args.opt_type
        self.optimizer_mode = args.opt_mode
        self.num_actions = args.num_actions
        self.initial_lr = args.initial_lr
        self.lr_annealing_steps = args.lr_annealing_steps

        # Shared mem vars
        self.learning_vars = args.learning_vars
        size = self.learning_vars.size
        self.flat_grads = np.empty(size, dtype=ctypes.c_float)

        if (self.optimizer_mode == 'local'):
            if (self.optimizer_type == 'rmsprop'):
                self.opt_st = np.ones(size, dtype=ctypes.c_float)
            else:
                self.opt_st = np.zeros(size, dtype=ctypes.c_float)
        elif (self.optimizer_mode == 'shared'):
            self.opt_st = args.opt_state

        self.e = args.e
        # rmsprop/momentum
        self.alpha = args.alpha
        # adam
        self.b1 = args.b1
        self.b2 = args.b2

        from atari_environment import AtariEnvironment
        self.emulator = AtariEnvironment(args.game, args.visualize)

        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma

        # Barrier to synchronize all actors after initialization is done
        self.barrier = args.barrier
        self.game = args.game

        # Initizlize Tensorboard summaries
        self.summary_ph, self.update_ops, self.summary_ops = self.setup_summaries()
        self.summary_op = tf.summary.merge_all()

    def synchronize_workers(self):
        if self.actor_id == 0:
            # Initialize network parameters
            g_step = utils.restore_vars(self.saver, self.session, self.game, self.max_local_steps)
            self.global_step.progress(g_step)
            self.last_saving_step = g_step
            logger.debug('T{}: Initializing shared memory...'.format(self.actor_id))
            self.init_shared_memory()

        # Wait until actor 0 finishes initializing shared memory
        self.barrier.wait()

        if self.actor_id > 0:
            logger.debug('T{}: Syncing with shared memory...'.format(self.actor_id))
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
        # Ensure we don't add any more nodes to the graph
        self.session.graph.finalize()

        # Wait until all actors are ready to start
        self.barrier.wait()

        # Introduce a different start delay for each actor, so that they do not run in synchronism.
        # This is to avoid concurrent updates of parameters as much as possible
        time.sleep(self.actor_id)

    @contextmanager
    def monitored_environment(self):
        """
        if self.use_monitor:
            self.log_dir = tempfile.mkdtemp()
            self.emulator.env = gym.wrappers.Monitor(self.emulator.env, self.log_dir)
        """
        yield
        self.emulator.env.close()

    def run(self):
        num_cpus = multiprocessing.cpu_count()

        self.supervisor = tf.train.Supervisor(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.global_variables_initializer(),
            logdir=self.summ_base_dir,
            saver=self.saver,
            summary_op=None)

        session_context = self.supervisor.managed_session(config=tf.ConfigProto(
            intra_op_parallelism_threads=num_cpus,
            inter_op_parallelism_threads=num_cpus,
            gpu_options=tf.GPUOptions(allow_growth=True),
            allow_soft_placement=True))

        with self.monitored_environment(), session_context as self.session:
            self.synchronize_workers()
            self.train()

    def save_vars(self):
        if (self.actor_id == 0 and
                (self.global_step.value() - self.last_saving_step >= CHECKPOINT_INTERVAL)):
            self.last_saving_step = self.global_step.value()
            utils.save_vars(self.saver, self.session, self.game, self.max_local_steps,
                            self.last_saving_step)

    def init_shared_memory(self):
        # Initialize shared memory with tensorflow var values
        params = self.session.run(self.local_network.params)
        # Merge all param matrices into a single 1-D array
        params = np.hstack([p.reshape(-1) for p in params])
        np.frombuffer(self.learning_vars.vars, ctypes.c_float)[:] = params

    def apply_gradients_to_shared_memory_vars(self, grads):
        # Flatten grads
        offset = 0
        for g in grads:
            self.flat_grads[offset:offset + g.size] = g.reshape(-1)
            offset += g.size
        g = self.flat_grads

        if self.optimizer_type == 'adam' and self.optimizer_mode == 'shared':
            p = np.frombuffer(self.learning_vars.vars, ctypes.c_float)
            p_size = self.learning_vars.size
            m = np.frombuffer(self.opt_st.ms, ctypes.c_float)
            v = np.frombuffer(self.opt_st.vs, ctypes.c_float)
            T = self.global_step.value()
            self.opt_st.lr.value = 1.0 * self.opt_st.lr.value * (1 - self.b2 ** T) ** 0.5 / (1 - self.b1 ** T)

            apply_grads_adam(m, v, g, p, p_size, self.opt_st.lr.value, self.b1, self.b2, self.e)

        else:  # local or shared rmsprop/momentum
            lr = self.decay_lr()
            if (self.optimizer_mode == 'local'):
                m = self.opt_st
            else:  # shared
                m = np.frombuffer(self.opt_st.vars, ctypes.c_float)

            p = np.frombuffer(self.learning_vars.vars, ctypes.c_float)
            p_size = self.learning_vars.size
            _type = 0 if self.optimizer_type == 'momentum' else 1

            apply_grads_mom_rmsprop(m, g, p, p_size, _type, lr, self.alpha, self.e)

    def sync_net_with_shared_memory(self, dest_net, shared_mem_vars):
        feed_dict = {}
        offset = 0
        params = np.frombuffer(shared_mem_vars.vars,
                               ctypes.c_float)
        for i in range(len(dest_net.params)):
            shape = shared_mem_vars.var_shapes[i]
            size = np.prod(shape)
            feed_dict[dest_net.params_ph[i]] = \
                params[offset:offset + size].reshape(shape)
            offset += size

        self.session.run(dest_net.sync_with_shared_memory,
                         feed_dict=feed_dict)

    def decay_lr(self):
        if self.global_step.value() <= self.lr_annealing_steps:
            return self.initial_lr - (self.global_step.value() * self.initial_lr / self.lr_annealing_steps)
        else:
            return 0.0

    def log_summary(self, *args):
        if self.actor_id == 0:
            feed_dict = {ph: val for ph, val in zip(self.summary_ph, args)}
            summaries = self.session.run(self.update_ops + [self.summary_op], feed_dict=feed_dict)[-1]
            self.supervisor.summary_computed(self.session, summaries, global_step=self.global_step.value())

    def setup_summaries(self):
        episode_reward = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='episode_reward')  # @lezhang.thu
        s1 = tf.summary.scalar('Episode_Reward_{}'.format(self.actor_id), episode_reward)
        summary_vars = [episode_reward]

        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        with tf.control_dependencies(update_ops):
            summary_ops = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_ops
