# A3C
from network import *


class PolicyVNetwork(Network):
    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)

        entropy_beta = 0.01 # tensorpack

        with tf.name_scope(self.name):
            self.adv_actor_ph = tf.placeholder(tf.float32, [None], name='advantage')
            self.ox = self.o4

            # Final actor layer
            self.wpi, self.bpi, self.output_layer_pi = self._softmax(
                'softmax_policy4', self.ox, self.num_actions)

            log_probs = tf.log(self.output_layer_pi + 1e-6)
            xentropy_loss = \
                tf.reduce_sum(
                    self.output_layer_pi * log_probs, name='xentropy_loss')

            # Final critic layer
            self.wv, self.bv, self.output_layer_v = self._fc(
                'fc_value4', self.ox, 1, activation='linear')

            self.params = [self.w1, self.b1, self.w2, self.b2, self.w3,
                           self.b3, self.w4, self.b4, self.wpi, self.bpi, self.wv, self.bv]

            # Actor objective
            # Multiply the output of the network by a one hot vector, 1 for the
            # executed action. This will make the non-regularised objective
            # term for non-selected actions to be zero.
            log_pi_a_given_s = \
                tf.reduce_sum(
                    log_probs * \
                    self.selected_action_ph, axis=1)
            policy_loss = -tf.reduce_sum(  # - sign!
                self.adv_actor_ph * log_pi_a_given_s, name='policy_loss')

            value_loss = \
                -tf.reduce_sum(tf.multiply(  # - sign!
                    self.adv_actor_ph,
                    tf.reshape(self.output_layer_v, [-1])))

            # Optimizer
            grads_no_entropy = tf.gradients(
                tf.truediv(policy_loss + value_loss,
                           tf.constant(conf['args'].max_local_steps, tf.float32)),  # @lezhang
                self.params)
            grads_entropy = tf.gradients(
                tf.truediv(xentropy_loss * entropy_beta,
                           tf.constant(conf['args'].max_local_steps, tf.float32)),
                self.params)
            # print('{}'.format(grads_entropy))

            # Placeholders for shared memory vars
            self.params_ph = []
            for p in self.params:
                self.params_ph.append(
                    tf.placeholder(tf.float32, shape=p.get_shape()))

            # Ops to sync net with shared memory vars
            self.sync_with_shared_memory = []
            for i in range(len(self.params)):
                self.sync_with_shared_memory.append(
                    self.params[i].assign(self.params_ph[i]))

            for i in range(len(self.params)):
                grads_no_entropy[i] = grads_no_entropy[i] + self.params_ph[i]
            self.grads_pure = grads_no_entropy

            for i in range(len(self.params)):
                # variables affected by grads_entropy is less in number than grads_no_entropy
                if grads_entropy[i] is None:
                    grads_entropy[i] = grads_no_entropy[i]
                else:
                    grads_entropy[i] = grads_entropy[i] + grads_no_entropy[i]
                # pass
            grads = grads_entropy

            # This is not really an operation, but a list of gradient Tensors
            # When calling run() on it, the value of those Tensors
            # (i.e., of the gradients) will be calculated
            # default: tf.clip_by_average_norm(grad, 0.1)
            self.get_gradients = [tf.clip_by_average_norm(
                g, 0.1) for g in grads]
