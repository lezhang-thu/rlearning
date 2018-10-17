# -*- coding: utf-8 -*-
from __future__ import absolute_import
import pickle
import argparse
import numpy as np
import tensorflow as tf

import skimage.io as io
import skimage.transform as transform

import networkrl as nw_policy
from actions import command2action, generate_bbox, crop_input

from os.path import join
import networksl as nw_evaluate

import math
import collections

num_actions = 14
virtual_loss = 3
c_puct = 5
num_virtual_threads = 4
num_simulations = 5
#num_virtual_threads=1
global_dtype = tf.float32

#although in r0.12, the checkpoint is stored in multiple files, you can restore it by
#using the common prefix, which is 'model.ckpt' in your case.
#https://stackoverflow.com/a/41087789
snapshot = 'model-spp-max/model-spp-max'
embedding_dim = 1000
ranking_loss = 'svm'

graph_evaluate = tf.Graph()
with graph_evaluate.as_default():
    #Only useful when loading Python 2 generated pickled files in Python 3, which 
    #includes npy/npz files containing object arrays.
    #Values other than ‘latin1’, ‘ASCII’, and ‘bytes’ are not allowed, as they can corrupt numerical data.
    #https://stackoverflow.com/a/47814305
    net_data = np.load('alexnet.npy', encoding='bytes').item()

    image_placeholder_evaluate = tf.placeholder(
        dtype=global_dtype, shape=[None, 227, 227, 3]
    )
    var_dict = nw_evaluate.get_variable_dict(net_data)
    SPP = True
    pooling = 'max'
    with tf.variable_scope('ranker') as scope:
        feature_vec = nw_evaluate.build_alexconvnet(
            image_placeholder_evaluate,
            var_dict,
            embedding_dim,
            SPP=SPP,
            pooling=pooling
        )
        score_func = nw_evaluate.score(feature_vec)

sess_evaluate = tf.Session(config=tf.ConfigProto(), graph=graph_evaluate)
#load pre-trained model
with sess_evaluate.as_default():
    with graph_evaluate.as_default():
        saver = tf.train.Saver(tf.global_variables())
        sess_evaluate.run(tf.global_variables_initializer())
        saver.restore(sess_evaluate, snapshot)

graph_policy = tf.Graph()
with graph_policy.as_default():
    with open('vfn_rl.pkl', 'rb') as f:
        var_dict = pickle.load(f)

    image_placeholder_policy = tf.placeholder(
        dtype=global_dtype, shape=[None, 227, 227, 3]
    )
    global_feature_placeholder = nw_policy.vfn_rl(
        image_placeholder_policy, var_dict
    )

    h_placeholder = tf.placeholder(dtype=global_dtype, shape=[None, 1024])
    c_placeholder = tf.placeholder(dtype=global_dtype, shape=[None, 1024])
    action, h, c = nw_policy.vfn_rl(
        image_placeholder_policy,
        var_dict,
        global_feature=global_feature_placeholder,
        h=h_placeholder,
        c=c_placeholder
    )
sess_policy = tf.Session(graph=graph_policy)


class Node:
    def __init__(
        self,
        crop_img,
        hidden_state,
        cell_state,
        ratio,
        terminal,
        parent=None,
        Nsa=None,
        Wsa=None,
        Qsa=None,
        Psa=None
    ):
        #state: None is an internal node, not None is a leaf
        self.state = {
            'crop_img': crop_img,
            'hidden_state': hidden_state,
            'cell_state': cell_state,
            'ratio': ratio,
            'terminal': terminal
        }
        if parent is not None:
            self.parent_info = {
                'parent': parent,
                'statistics': {
                    'Nsa': Nsa,
                    'Wsa': Wsa,
                    'Qsa': Qsa,
                    'Psa': Psa
                }
            }
        else:
            self.parent_info = None
        self.children = None


class MCTS:
    def __init__(
        self,
        original_image,
        sess_policy,
        sess_evaluate,
        c_puct,
        num_virtual_threads=0,
        virtual_loss=0
    ):
        #no resize is useful for generate_bbox and crop_input
        self.original_image = np.expand_dims(
            original_image, axis=0
        )  #no resize!!!

        #the neural networks require the resized image!!!
        original_image = transform.resize(
            original_image, (227, 227), mode='constant'
        )  #after resize
        original_image = np.expand_dims(original_image, axis=0)

        with sess_policy.as_default():
            self.global_feature = sess_policy.run(
                global_feature_placeholder,
                feed_dict={
                    image_placeholder_policy: original_image
                }
            )

        self.root = Node(
            crop_img=original_image,
            hidden_state=np.zeros([1, 1024]),
            cell_state=np.zeros([1, 1024]),
            ratio=np.repeat([[0, 0, 20, 20]], 1, axis=0),
            terminal=np.zeros(1, dtype=bool)
        )

        self.param = {
            'c_puct': c_puct,
            'num_virtual_threads': num_virtual_threads,
            'virtual_loss': virtual_loss
        }
        with sess_evaluate.as_default():
            self.baseline = sess_evaluate.run(
                score_func,
                feed_dict={
                    image_placeholder_evaluate: original_image
                }
            )[0]
        #debug
        print('self.baseline {}'.format(self.baseline))
        self.sess = {'policy': sess_policy, 'evaluate': sess_evaluate}

    def _get_evaluation(self, crop_img):
        with self.sess['evaluate'].as_default():
            return self.sess['evaluate'].run(
                score_func, feed_dict={
                    image_placeholder_evaluate: crop_img
                }
            )[0] - self.baseline

    def _get_action_prob(self, crop_img, hidden_state, cell_state):
        with self.sess['policy'].as_default():
            return self.sess['policy'].run(
                (action, h, c),
                feed_dict={
                    image_placeholder_policy: crop_img,
                    global_feature_placeholder: self.global_feature,
                    h_placeholder: hidden_state,
                    c_placeholder: cell_state
                }
            )

    def simulate_many_times(self, num_times):
        assert num_times > 1

        for _ in range(num_times):
            self._simulate_once()

    def _simulate_once(self):
        leaf_ls = []
        for _ in range(self.param['num_virtual_threads']):
            leaf_ls.append(self._select())

#        visit_times = collections.Counter(leaf_ls)
#         #debug
#         print(visit_times)

        for leaf in leaf_ls:
            self._backup(leaf, self._rollout(leaf))
        leaf_set = set(leaf_ls)
        for leaf in leaf_set:
            self._expand(leaf)
            if not leaf.state['terminal'][0]:
                leaf.state = None
#         value_ls = []
#         for leaf in visit_times:
#             value_ls.append(self._rollout(leaf))
#             import sys
#             inp =input('continue? (y/n)')
#             if inp == 'y':
#                 raise KeyboardInterrupt
#             self._expand(leaf)
#             if not leaf.state['terminal'][0]:
#                 leaf.state = None  #leaf is updated to an internal node, if it is not of terminal state
#debug
#         print('value_ls {}'.format(value_ls))
#         for i, (leaf, freq) in enumerate(visit_times.items()):
#             self._backup(leaf, freq, value_ls[i])

    def _select(self):
        s = self.root
        while s.state is None:
            visit_total = 0
            for c in s.children:
                visit_total += c.parent_info['statistics']['Nsa']
            max_child = None
            for c in s.children:
                if max_child is None or (
                    max_child.parent_info['statistics']['Qsa'] +
                    self.param['c_puct'] * max_child.parent_info['statistics']
                    ['Psa'] * math.sqrt(visit_total) /
                    (1 + max_child.parent_info['statistics']['Nsa'])
                ) < (
                    c.parent_info['statistics']['Qsa'] +
                    self.param['c_puct'] * c.parent_info['statistics']['Psa'] *
                    math.sqrt(visit_total) /
                    (1 + c.parent_info['statistics']['Nsa'])
                ):
                    max_child = c
            s = max_child
            #'the rollout statistics are updated as if it has lost n_vl games'
            s.parent_info['statistics']['Nsa'] += self.param['virtual_loss']
            s.parent_info['statistics']['Wsa'] -= self.param['virtual_loss']
            s.parent_info['statistics']['Qsa'] = s.parent_info['statistics'][
                'Wsa'
            ] / s.parent_info['statistics']['Nsa']
        return s

    def _rollout(self, leaf):
        #notice the np.copy
        crop_img, hidden_state, cell_state, ratio, terminal = (
            leaf.state['crop_img'], leaf.state['hidden_state'],
            leaf.state['cell_state'], np.copy(leaf.state['ratio']),
            np.copy(leaf.state['terminal'])
        )
        #debug
        action_sample_ls = []
        while not terminal[0]:
            #sample an action
            action_prob, hidden_state, cell_state = self._get_action_prob(
                crop_img, hidden_state, cell_state
            )

            action_sample = np.random.choice(
                num_actions, 1, p=np.reshape(action_prob, (-1))
            )
            action_sample_ls.append(action_sample[0])
            ratio, terminal = command2action(action_sample, ratio, terminal)
            #debug
            #            print('ratio {}'.format(ratio))
            if not terminal[0]:
                #                print('run ...')
                bbox = generate_bbox(self.original_image, ratio)
                crop_img = crop_input(self.original_image, bbox)
        #debug


#        print('action_sample_ls {}'.format(action_sample_ls))
#reach a terminal state, then compute ranking scores
        return self._get_evaluation(crop_img)

    def _expand(self, leaf):
        if leaf.state['terminal'][0]:
            return
        action_prob, hidden_state, cell_state = self._get_action_prob(
            leaf.state['crop_img'], leaf.state['hidden_state'],
            leaf.state['cell_state']
        )
        action_prob = np.reshape(action_prob, (-1))

        action_ls = np.array(range(num_actions), dtype=np.int32)
        ratio_ls = np.repeat(
            leaf.state['ratio'], num_actions, axis=0
        )  #intial ratio is of batch_size 1
        terminal_ls = np.repeat(
            leaf.state['terminal'], num_actions, axis=0
        )  #intial terminal is of batch_size 1
        ratio_ls, terminal_ls = command2action(action_ls, ratio_ls, terminal_ls)
        bbox_ls = generate_bbox(
            np.repeat(self.original_image, num_actions, axis=0), ratio_ls
        )
        crop_img_ls = crop_input(
            np.repeat(self.original_image, num_actions, axis=0), bbox_ls
        )

        leaf.children = []
        for i in range(num_actions):
            leaf.children.append(
                Node(
                    crop_img=np.expand_dims(crop_img_ls[i], axis=0),
                    hidden_state=hidden_state,
                    cell_state=cell_state,
                    ratio=np.expand_dims(ratio_ls[i], axis=0),
                    terminal=np.expand_dims(terminal_ls[i], axis=0),
                    parent=leaf,
                    Nsa=0,
                    Wsa=0,
                    Qsa=0,
                    Psa=action_prob[i]
                )
            )

    def _backup(self, leaf, value):
        s = leaf
        while s.parent_info is not None:
            s.parent_info['statistics']['Nsa'
                                       ] += (-self.param['virtual_loss'] + 1)
            s.parent_info['statistics'
                         ]['Wsa'] += (self.param['virtual_loss'] + value)
            s.parent_info['statistics']['Qsa'] = s.parent_info['statistics'][
                'Wsa'
            ] / s.parent_info['statistics']['Nsa']
            s = s.parent_info['parent']

    def get_crop_box(self):
        if self.root.state is None:  #require a leaf node
            return None
        return generate_bbox(self.original_image, self.root.state['ratio'])[0]

    def reach_terminal(self):
        if self.root.state is not None and self.root.state['terminal'][0]:
            return True
        return False

    def act_one_step(self):
        #select the action with maximum visit count
        if self.root.state is not None:  #a leaf node
            return -1

        #root is an internal node
        max_child = None
        action = -1
        for i, c in enumerate(self.root.children):
            if max_child is None or max_child.parent_info['statistics'][
                'Nsa'
            ] < c.parent_info['statistics']['Nsa']:
                max_child = c
                action = i
        self.root = max_child
        self.root.parent_info = None
        return action

    def debug_print(self):
        if self.root.state is not None:
            print('reach a leaf')
        else:
            for i, c in enumerate(self.root.children):
                print(
                    'Nsa {}, Wsa {}, Qsa {}, Psa {}'.format(
                        c.parent_info['statistics']['Nsa'],
                        c.parent_info['statistics']['Wsa'],
                        c.parent_info['statistics']['Qsa'],
                        c.parent_info['statistics']['Psa']
                    )
                )
                print('above info for child {}'.format(i))
            print('end of the printing this time ...')


def auto_cropping(original_image):
    mcts_tree = MCTS(
        original_image, sess_policy, sess_evaluate, c_puct, num_virtual_threads,
        virtual_loss
    )
    action_ls = []
    while not mcts_tree.reach_terminal():
        mcts_tree.simulate_many_times(num_simulations)
        #        mcts_tree.debug_print()
        action_ls.append(mcts_tree.act_one_step())


#     mcts_tree.simulate_once()
#     mcts_tree.simulate_once()
#     mcts_tree.debug_print()
#     action_ls.append(mcts_tree.act_one_step())
    return action_ls, mcts_tree.get_crop_box()

if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Auto Image Cropping')
    #parser.add_argument('--image_path', required=True, help='Path for the image to be cropped')
    #parser.add_argument('--save_path', required=True, help='Path for saving cropped image')

    #args = parser.parse_args()
    image_path = join('RL_test_images', '639741.jpg')
    save_path = join('RL_test_images', '639741_cropped.jpg')
    im = io.imread(image_path).astype(np.float32) / 255
    action_ls, (xmin, ymin, xmax, ymax) = auto_cropping(im - 0.5)
    print('action list {}\n'.format(action_ls))
    io.imsave(save_path, im[ymin:ymax, xmin:xmax])
