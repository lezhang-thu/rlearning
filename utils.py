import tensorflow as tf
import os


def restore_vars(saver, sess, game, max_local_steps):
    """ Restore saved net, global step, and epsilons OR 
    create checkpoint directory for later storage. """

    alg = '{}/'.format('a3c_' + str(max_local_steps))
    checkpoint_dir = 'train_data/checkpoints/' + game + '/' + alg

    check_or_create_checkpoint_dir(checkpoint_dir)
    path = tf.train.latest_checkpoint(checkpoint_dir)
    if path is None:
        return 0
    else:
        saver.restore(sess, path)
        global_step = int(path[path.rfind('-') + 1:])
        return global_step


def save_vars(saver, sess, game, alg_type, max_local_steps, global_step):
    """ Checkpoint shared net params, global score and step, and epsilons. """

    alg = '{}/'.format('a3c_' + str(max_local_steps))
    checkpoint_dir = 'train_data/checkpoints/' + game + '/' + alg

    check_or_create_checkpoint_dir(checkpoint_dir)
    saver.save(sess, checkpoint_dir + 'model', global_step=global_step)


def check_or_create_checkpoint_dir(checkpoint_dir):
    """ Create checkpoint directory if it does not exist """
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass
