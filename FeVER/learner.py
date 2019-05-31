# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2017-04-03 13:22:46
# Last modified: 2019-04-05 09:54:09

"""
The training algorithm for FeVER system.

In order to separte the model and algorihtm, I decided to split
the training algorithm from the model.py
"""

import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm
import ExAssist as EA

import FeVER.utils as utils
from FeVER import Logger
from FeVER.loss import LossFuntion


class Learner:
    """Algorithm class implements algorithm to train the model.
    This algorihtm is rely on GPU version of Pytorch.
    """
    def __init__(self, args):
        """
        Initialize the algorithm instance.

        args:
            parameters: dict - The parameters of this model.
        """
        self._args = args

    def train(self, dataset, model):
        """Train the model based on the given (x, Y) pairs.
        """
        args = self._args
        logger = logging.getLogger(Logger.project_name)

        self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.rate)

        # Prepare loss fucntion
        s = 'Preparing the loss functions...'
        logger.info(s)
        loss = LossFuntion(args)
        floss = self._init_train(loss)

        training_start = time.perf_counter()
        logger.info('Start training.')
        assist = EA.getAssist('GEL')

        for epoch in range(args.max_iter):
            s = '{a}-th epoch of training...'.format(a=str(epoch))
            logger.info(s)

            start = time.perf_counter()
            allloss = self._train(dataset, floss, model)
            end = time.perf_counter()

            s = 'Finish {a}-th epoch, loss={c}, time={b} s'.format(
                a=str(epoch),  b=str(end-start), c=str(allloss))
            logger.info(s)
            assist.info['loss'] = str(allloss)
            assist.step()
            if (epoch + 1) % 10 == 0:
                path = args.model_file + '.' + str(epoch+1)
                torch.save(model, path)

        training_end = time.perf_counter()
        s = 'End training process, loss={a}, totoal time={b}'.format(
            a=str(allloss), b=str(training_end-training_start))
        logger.info(s)

    ########################################################
    # Private methods
    ########################################################
    def _init_train(self, loss):
        """Init the training process, return
        two functions:
            floss: Loss function.
            fprepare: The prepare function.
        """
        args = self._args

        if args.modeling == 1:
            floss = loss.local_hinge
        elif args.modeling == 2:
            floss = loss.global_hinge
        elif args.modeling == 3:
            floss = loss.local_log
        elif args.modeling == 4:
            floss = loss.global_log
        else:
            raise Exception('Undefined modeling')
        return floss

    def _train(self, data, floss, model):
        """Train each epoch.
        Args:
            data: iterator
        """
        allloss = 0
        args = self._args
        for batch in data:
            batch = utils.batch2GPU(batch, args)
            self.optimizer.zero_grad()
            # Score current batch
            x, xoffsets = batch[0][0], batch[0][1]
            y, yoffsets = batch[-1][0], batch[-1][1]
            scores = model(x, y, xoffsets, yoffsets)
            # Compute the loss
            loss = floss(scores, batch)
            loss = self._regularize(loss, model)
            t = loss.data.cpu().numpy()
            allloss += t
            loss.backward()
            self.optimizer.step()

            if args.W_reg == 'OnBall':
                model.W.weight.data = nn.functional.normalize(
                        model.W.weight.data)
            if args.V_reg == 'OnBall':
                model.V.weight.data = nn.functional.normalize(
                        model.V.weight.data)

        return allloss

    def _regularize(self, loss, model):
        args = self._args
        if args.W_reg == 'L2':
            loss += self._L2regularizer(model.W.weight, args.WC)
        if args.V_reg == 'L2':
            loss += self._L2regularizer(model.V.weight, args.VC)
        return loss

    def _L2regularizer(self, M, C):
        """Calculate the L2 regularizer
        on given M matrix.

        Args:
            M: Variable
            C: tradoff

        Returns:
            loss: Variable - The loss.
        """
        return C * (torch.norm(M, 2) ** 2)
