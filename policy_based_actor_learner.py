# A3C
import pickle
from actor_learner import *
from policy_v_network import *
import pyximport

pyximport.install()
from fast_cts import CTSDensityModel


class A3CLearner(ActorLearner):
    def __init__(self, args):
        super(A3CLearner, self).__init__(args)

        # Shared mem vars
        self.learning_vars = args.learning_vars
        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'num_act': self.num_actions,
                         'args': args}
        self.local_network = PolicyVNetwork(conf_learning)

        # @lezhang
        self.e_tm1 = []  # e_t-1
        for i in range(len(self.learning_vars.var_shapes)):
            shape = self.learning_vars.var_shapes[i]
            self.e_tm1.append(np.zeros(shape, np.float32))
        self.batch_size = self.max_local_steps
        self.param_lambda = 0.9  # @lezhang.thu

        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(
                var_list=var_list, max_to_keep=3,
                keep_checkpoint_every_n_hours=2)

        # @lezhang
        self.cts = CTSDensityModel()
        self.cts_updated = args.cts_updated
        self.cts_lock = args.cts_lock
        self.cts_sync_steps = args.cts_sync_steps

    def write_cts(self):
        with self.cts_lock:
            if self.cts_updated[self.actor_id] == 1:
                return
            raw_data = pickle.dumps(self.cts.get_state())
            with open('train_data/cts_data.pkl', 'wb') as f:
                f.write(raw_data)
            for i in range(len(self.cts_updated)):
                self.cts_updated[i] = 1

    def read_cts(self):
        with open('train_data/cts_data.pkl', 'rb') as f:
            raw_data = f.read()
        self.cts.set_state(pickle.loads(raw_data))
        self.cts_updated[self.actor_id] = 0

    def choose_next_action(self, state):
        new_action = np.zeros([self.num_actions])
        value, distrib = \
            self.session.run([
                self.local_network.output_layer_v,
                self.local_network.output_layer_pi],
                feed_dict={self.local_network.input_ph: [state]})

        distrib = distrib.reshape(-1)
        value = np.asscalar(value)

        assert np.all(np.isfinite(distrib)), distrib
        action = np.random.choice(len(distrib), p=distrib)
        new_action[action] = 1

        return new_action, value

    def train(self):
        """ Main actor learner loop for advantage actor critic learning. """
        s = self.emulator.get_initial_state()
        episode_reward = 0.0

        z = 1
        gamma_lambda_exps = []
        for _ in range(self.batch_size):
            gamma_lambda_exps.append(z)
            z *= self.gamma * self.param_lambda
        gamma_lambda_exps.reverse()  # z=(\gamma*\lambda)^batch_size

        episode_over = False
        rewards = []
        states = []
        actions = []
        values = []
        adv_batch = []

        while self.global_step.value() < self.max_global_steps:

            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)

            if self.local_step % self.cts_sync_steps == 0:
                self.write_cts()
                self.read_cts()
            local_step_start = self.local_step

            new_s = None
            while not (episode_over
                       or (self.local_step - local_step_start
                               == self.batch_size)):
                # Choose next action and execute it
                a, v = self.choose_next_action(s)

                new_s, reward, episode_over = self.emulator.next(a)
                episode_reward += reward
                reward += float(self.cts.update(new_s[..., -1]))

                # (S_t, A_t, \hat{V}(S_t, \theta), R_t+1)
                rewards.append(np.clip(reward, -1, 1))
                states.append(s)
                actions.append(a)
                values.append(v)

                s = new_s
                self.local_step += 1

            # Calculate the value offered by critic in the new state.
            if episode_over:
                values.append(0.0)
            else:
                values.append(
                    self.session.run(
                        self.local_network.output_layer_v,
                        feed_dict={self.local_network.input_ph: [new_s]})[0][0])

            G_lambda_t_i = values[-1]
            for i in reversed(range(len(states))):
                G_lamdba_t_i = \
                    rewards[i] + self.gamma * \
                                 (1 - self.param_lambda) * values[i + 1] + \
                    self.gamma * self.param_lambda * G_lambda_t_i
                adv_batch.append(G_lambda_t_i - values[i])
            adv_batch.reverse()

            x = 0
            y = 1
            for j in range(len(states)):
                y *= self.gamma * self.param_lambda
                x += (rewards[j] + self.gamma * values[j + 1] - values[j]) * y

            # Compute gradients on the local policy/V network and apply them to shared memory
            feed_dict = {
                self.local_network.input_ph: states,
                self.local_network.selected_action_ph: actions,
                self.local_network.adv_actor_ph: adv_batch}
            for i in range(len(self.local_network.params)):
                feed_dict[self.local_network.params_ph[i]] = \
                    self.e_tm1[i] * x

            grads = self.session.run(
                self.local_network.get_gradients,
                feed_dict=feed_dict)

            if not episode_over:
                feed_dict[self.local_network.adv_actor_ph] = \
                    gamma_lambda_exps

                for i in range(len(self.local_network.params)):
                    feed_dict[self.local_network.params_ph[i]] = \
                        self.e_tm1[i] * z
                self.e_tm1 = self.session.run(
                    self.local_network.grads_pure,
                    feed_dict=feed_dict)
            else:
                for i in range(len(self.e_tm1)):
                    self.e_tm1[i] = np.zeros_like(self.e_tm1[i])

            self.apply_gradients_to_shared_memory_vars(grads)

            del rewards[:]
            del states[:]
            del actions[:]
            del values[:]
            del adv_batch[:]
            self.global_step.progress(self.local_step - local_step_start)

            # Start a new game on reaching terminal state
            if episode_over:
                logger.debug('T{}, Step {}, Reward {}'.format(self.actor_id, self.global_step.value(), episode_reward))
                self.log_summary(episode_reward)

                s = self.emulator.get_initial_state()
                episode_over = False
                episode_reward = 0.0

                # @lezhang
                # self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
