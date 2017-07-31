#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pseudocount.py
# Author: Music Li <yuezhanl@andrew.cmu.edu>
from __future__ import division
from tensorpack.RL.cts_models import (ConvolutionalMarginalDensityModel, ConvolutionalDensityModel, 
    L_shaped_context, LocationDependentDensityModel)
import cv2
import numpy as np
FRSIZE = 42
MAXVAL = 255                # original max value for a state
MAX_DOWNSAMPLED_VAL = None   # downsampled max value for a state. 8 in the paper.
FEATURE_NUM = 512
from collections import defaultdict

__all__ = ['PSC']

class PSC():
    # class for process with pseudo count rewards
    def __init__(self, method, downsample_value=32, UCB1=False):
        # initialize
        self.method = method
        self.UCB1 = UCB1
        
        global MAX_DOWNSAMPLED_VAL

        if self.method == 'CTS':
            MAX_DOWNSAMPLED_VAL = 8
            #self.CTS = ConvolutionalMarginalDensityModel((FRSIZE, FRSIZE))           
            # 100 iter/s for memory filling
            #self.CTS = ConvolutionalDensityModel((FRSIZE, FRSIZE), L_shaped_context) 
            # 12 iter/s for memory filling
            self.CTS = LocationDependentDensityModel((FRSIZE, FRSIZE), L_shaped_context) # 12 iter/s
        else:
            MAX_DOWNSAMPLED_VAL = downsample_value
        
        self.flat_pixel_counter = np.zeros((FRSIZE*FRSIZE, 
            MAX_DOWNSAMPLED_VAL+1)) # Counter for each (pos1, pos2, val), used for joint method
        self.flat_feature_counter = np.zeros((FEATURE_NUM, MAX_DOWNSAMPLED_VAL + 1))
        self.total_num_states = 0  # total number of seen states

        self.n = 0
        self.dict = defaultdict(int)

    def clear(self):
        self.flat_pixel_counter = np.zeros((FRSIZE*FRSIZE, MAX_DOWNSAMPLED_VAL+1))
        self.total_num_states = 0
        return

    def psc_reward(self, state):
        """
        The final API used by others.
        Given an state, return back the final pseudo count reward
        :return:
        """
        state = self.preprocess(state)
        pc_reward = self.add(state)

        return pc_reward

    def pc_reward_with_act(self, state, act):
        state = self.preprocess(state)
        # Brute force idea
        sa_hash = hash(state.tostring()+str(act))
        count = self.dict[sa_hash]
        self.dict[sa_hash] += 1
        return self.count2reward(count)

    def pc_reward_feature(self, feature):
        # scale feature to 0 - 1
        feature = self.standardize_feature(feature)
        # discretize features
        feature = self.discretize_feature(feature)
        # print feature
        if self.method == 'joint':
            # Model each pixel as independent pixels.
            # p = (c1/n) * (c2/n) * ... * (cn/n)
            # pp = (c1+1)/(n+1) * (c2+1)/(n+1) ...
            # N = (p/pp * (1-pp))/(1-p/pp) ~= (p/pp) / (1-p/pp)
            state = feature
            if self.total_num_states > 0:
                nr = (self.total_num_states + 1.0) / self.total_num_states
                pixel_count = self.flat_feature_counter[range(FEATURE_NUM), state]
                self.flat_feature_counter[range(FEATURE_NUM), state] += 1
                p_over_pp = np.prod(nr * pixel_count / (1.0 + pixel_count))
                denominator = 1.0 - p_over_pp
                if denominator <= 0.0:
                    # print "psc_add_image: dominator <= 0.0 : dominator=", denominator
                    denominator = 1.0e-20
                pc_count = p_over_pp / denominator
                pc_reward = self.count2reward(pc_count)
            else:
                pc_count = 0.0
                pc_reward = self.count2reward(pc_count)
            self.total_num_states += 1
            return pc_reward

    def standardize_feature(self, feature):
        min_feature = np.min(feature)
        max_feature = np.max(feature)
        feature = (feature - min_feature) / (max_feature - min_feature)
        return feature

    def discretize_feature(self, feature):
        feature = np.asarray(MAX_DOWNSAMPLED_VAL * feature, dtype=int)
        return feature

    def preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (FRSIZE, FRSIZE))
        state = np.uint8(state / MAXVAL * MAX_DOWNSAMPLED_VAL)
        return state

    def add(self, state):
        self.n += 1
        if self.method == 'joint':
            # Model each pixel as independent pixels.
            # p = (c1/n) * (c2/n) * ... * (cn/n)
            # pp = (c1+1)/(n+1) * (c2+1)/(n+1) ...
            # N = (p/pp * (1-pp))/(1-p/pp) ~= (p/pp) / (1-p/pp)
            state = np.reshape(state, (FRSIZE*FRSIZE))
            if self.total_num_states > 0:
                nr = (self.total_num_states + 1.0) / self.total_num_states
                pixel_count = self.flat_pixel_counter[range(FRSIZE*FRSIZE), state]
                self.flat_pixel_counter[range(FRSIZE*FRSIZE), state] += 1
                p_over_pp = np.prod(nr*pixel_count/(1.0+pixel_count))
                denominator = 1.0 - p_over_pp
                if denominator <= 0.0:
                    # print "psc_add_image: dominator <= 0.0 : dominator=", denominator
                    denominator = 1.0e-20
                pc_count = p_over_pp / denominator
                pc_reward = self.count2reward(pc_count)
            else:
                pc_count = 0.0
                pc_reward = self.count2reward(pc_count)
            self.total_num_states += 1
            return pc_reward
        if self.method == 'CTS':
            # Model described in the paper "Unifying Count-Based Exploration and Intrinsic Motivation"
            log_p = self.CTS.update(state)
            log_pp = self.CTS.query(state)
            n = self.p_pp_to_count(log_p, log_pp)
            pc_reward = self.count2reward(n)
            ## Following codes are used for generating images during training for debug
            # if self.n == 200:
            #     import matplotlib.pyplot as plt
            #     img = self.CTS.sample()
            #     plt.imshow(img)
            #     plt.show()
            return pc_reward

    def p_pp_to_count(self, log_p, log_pp):
        """
        :param p: density estimation p. p = p(x;x_{<t})
        :param pp: recoding probability. p' = p(x;x_{<t}x)
        :return: N = p(1-pp)/(pp-p) = (1-pp)/(pp/p-1) ~= 1/(pp/p-1)
        pp/p = e^(log_pp) / e^(log_p) = e ^ (log_pp - log_p)
        """
        #assert log_pp >= log_p
        pp = np.exp(log_pp)
        #assert pp <= 1
        pp_over_p = np.exp(log_pp - log_p)
        N = (1.0-pp) / (pp_over_p - 1)
        return N


    def count2reward(self, count, beta=0.05, alpha=0.01, power=-0.5):
        # r = beta (N + alpha)^power
        if not self.UCB1:
            reward = beta * ((count+alpha) ** power)
        else:
            #reward = beta * ((2 * np.log(self.total_num_states) / (count + alpha)) ** (-power))
            reward = 0.05 * ((count + alpha) ** power)
        return reward
