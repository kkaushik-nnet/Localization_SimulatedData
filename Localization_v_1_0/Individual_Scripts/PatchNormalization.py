#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Performs patch/receptive field normalization
#
# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
#
#

from mdp import *
import cv2
from mdp import Node, numx, NodeException, TrainingException

class NormalizePatchNode(Node):
    def __init__(self, input_dim=None, dtype=None):
        super(NormalizePatchNode, self).__init__(input_dim=input_dim, output_dim=input_dim, dtype=dtype)
        self.eps = 1e-5

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _execute(self, x):
        x_ = x.copy()
        x_ -= numx.atleast_2d(x_.mean(axis=1)).T
        x_ /= numx.atleast_2d(x_.std(axis=1)).T + numx.random.random(x_.shape) / 100.  # avoid division by zero
        return x_
