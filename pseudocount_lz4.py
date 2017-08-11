#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pseudocount_lz4.py
# Author: Music Li <yuezhanl@andrew.cmu.edu>
from __future__ import division

import cv2
import numpy as np

__all__ = ['PSC']


class PSC(object):
    def __init__(self, image_shape=None, color_max=None):
        # initialize
        self.color_max = color_max
        self.image_shape = image_shape
        self.flatten_size = np.prod(self.image_shape)
        # counter for each (pos1, pos2, val), used for joint method
        self.flat_pixel_counter \
            = np.zeros((self.flatten_size, self.color_max), dtype=np.float32)
        self.total_num_states = 0  # total number of seen states

    def psc_reward(self, image):
        state = self._preprocess(image)
        bonus = self._update(state)
        return bonus

    def get_state(self):
        return self.total_num_states, self.flat_pixel_counter

    def set_state(self, state):
        self.total_num_states, self.flat_pixel_counter = state

    def _preprocess(self, image):
        state = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.image_shape)
        return state.astype(np.uint8)

    def _update(self, state):
        """
        model pixels as independent.
        p = c1/n * c2/n * ... * cn/n
        pp = (c1+1)/(n+1) * ... * (cn+1)/(n+1)
        """
        state = np.reshape(state, self.flatten_size)
        if self.total_num_states == 0:
            psc_count = 0.0
        else:
            ratio = self.total_num_states / (self.total_num_states + 1.0)
            ratio_np1 = 1.0 / (self.total_num_states + 1.0)
            pixel_count = self.flat_pixel_counter[range(self.flatten_size), state]
            pp = np.prod(ratio_np1 * (pixel_count + 1.0))
            pp_over_p = np.prod(ratio * ((1.0 + pixel_count) / pixel_count))
            psc_count = (1.0 - pp) / np.maximum(pp_over_p - 1.0, 1e-8)
        self.flat_pixel_counter[range(self.flatten_size), state] += 1.0
        self.total_num_states += 1

        psc_reward = self._count2reward(psc_count)
        return psc_reward

    def _count2reward(self, count, beta=0.05, alpha=0.01, power=-0.5):
        return beta * ((count + alpha) ** power)
