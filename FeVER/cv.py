# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-10-30 09:24:21
# Last modified: 2019-04-05 09:53:26

"""
CV process for the selecting the best hyper-parameters.
"""
import logging

import torch

from FeVER.dataset import XMLDataset
from FeVER import Logger
from FeVER.learner import Learner
from FeVER.model import Model
from FeVER.eval import Eval
import FeVER.utils as utils


class CV:
    def __init__(self, args):
        logger = logging.getLogger(Logger.project_name)
        s = 'Initializing the CV process...'
        logger.info(s)
        self._args = args
        dataset = XMLDataset(args)
        args.label_feat_num = dataset.label_feat_num
        args.input_feat_num = dataset.input_feat_num
        args.label_num = dataset.label_num
        self.dataset = dataset
        self.evalator = Eval()

    def runcv(self):
        logger = logging.getLogger(Logger.project_name)
        args = self._args
        dataset = self.dataset
        cv_rates = args.cv_rates
        results = []
        args.max_iter = args.cv_max_iter
        for rate in cv_rates:
            s = 'rate={a}'.format(a=str(rate))
            logger.info(s)
            args.rate = rate
            model = Model(args)
            learner = Learner(args)
            learner.train(dataset, model)
            dev_data = dataset.dev_data
            label_feat = dataset.label_feat
            x, xoffsets = dev_data[0], dev_data[1]
            y, yoffsets = label_feat[0], label_feat[1]
            with torch.no_grad():
                scores = model(x, y, xoffsets, yoffsets)
            t = self.evalator.eval(
                    scores, dataset.dev_labels, 1)
            results.append((rate, t))
            s = 'rate={a}, top5={b}'.format(a=str(rate),
                                            b=str(t))
            logger.info(s)
            del model
            del learner
            torch.cuda.empty_cache()

        utils.write_cv(results, args.cv_output_file)
