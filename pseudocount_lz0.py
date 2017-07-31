#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pseudocount_lz0.py
# Author: Music Li <yuezhanl@andrew.cmu.edu>, Michael <lezhang.thu@gmail.com>
from __future__ import division

import cv2
import numpy as np

FRSIZE = 42
MAXVAL = 255  # original max value for a state
MAX_DOWNSAMPLED_VAL = None  # downsampled max value for a state

__all__ = ['PSC']


class PSC():
    def __init__(self, downsampled_value=128):
        # initialize
        global MAX_DOWNSAMPLED_VAL
        MAX_DOWNSAMPLED_VAL = downsampled_value

        # Counter for each (pos1, pos2, val), used for joint method
        self.flat_pixel_counter \
            = np.zeros((FRSIZE * FRSIZE, MAX_DOWNSAMPLED_VAL + 1), dtype=np.float32)
        self.total_num_states = 0  # total number of seen states

    def psc_reward(self, state):
        state = self._preprocess(state)
        bonus = self._update(state)
        return bonus

    def get_state(self):
        return self.total_num_states, self.flat_pixel_counter

    def set_state(self, state):
        self.total_num_states, self.flat_pixel_counter = state

    def _preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (FRSIZE, FRSIZE))
        state = np.uint8(state / MAXVAL * MAX_DOWNSAMPLED_VAL)
        return state

    def _update(self, state):
        """
        Model each pixel as independent pixels.
        p = c1/n * c2/n * ... * cn/n
        pp = (c1+1)/(n+1) * ... * (cn+1)/(n+1)
        """
        state = np.reshape(state, (FRSIZE * FRSIZE))
        if self.total_num_states == 0:
            psc_count = 0.0
        else:
            ratio = self.total_num_states / (self.total_num_states + 1.0)
            ratio_np1 = 1.0 / (self.total_num_states + 1.0)
            pixel_count = self.flat_pixel_counter[range(FRSIZE * FRSIZE), state]
            pp = np.prod(ratio_np1 * (pixel_count + 1.0))
            pp_over_p = np.prod(ratio * (1.0 + pixel_count) / pixel_count)
            psc_count = (1.0 - pp) / np.maximum(pp_over_p - 1.0, 1e-8)
        self.flat_pixel_counter[range(FRSIZE * FRSIZE), state] += 1.0
        self.total_num_states += 1

        psc_reward = self._count2reward(psc_count)
        return psc_reward

    def _count2reward(self, count, beta=0.05, alpha=0.01, power=-0.5):
        return beta * ((count + alpha) ** power)
