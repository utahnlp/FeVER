# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.5.2
#
# Date: 2017-06-07 10:23:18
# Last modified: 2019-04-05 09:30:57

"""
Read configuration for FeVER.
"""

import torch


class Config:
    """Config class for FeVER."""
    def __init__(self, config):
        self._config = config
        self._load()

    def _load(self):
        common = self._config['common']
        self._get_common(common)

        train = self._config['train']
        self._get_train_config(train)

        gen = self._config['generation']
        self._get_gen_config(gen)

        cv = self._config['cv']
        self._get_cv(cv)

        runpath = self._config['run']
        self._get_runpath(runpath)

    def _get_gen_config(self, config):
        self.embedding = config.get('embedding')
        self.pred_model_file = config.get('pred_model_path')
        self.pred_output_file = config.get('pred_output_path')
        self.pred_feat_file = config.get('pred_feat_path')
        self.pred_dict_file = config.get('pred_dict_path')

    def _get_runpath(self, config):
        self.model_file = config.get('model_file')
        self.label_embed_file = config.get('label_embed_path')
        self.ctx_embed_file = config.get('ctx_embed_path')
        self.cv_output_file = config.get('cv_output')

    def _get_common(self, config):
        self.mode = config.get('mode')
        self.num_workers = config.getint('num_workers')
        self.enable_cuda = config.getboolean('enable_cuda')
        self.cuda = self.enable_cuda and torch.cuda.is_available()

        # Training files
        self.train_file = config.get('train_file')
        self.label_file = config.get('label_file')
        self.counts_file = config.get('counts_file', fallback=None)
        self.weight_file = self.train_file + '.weight'

        # Context fetures
        self.gen_ctx_embedding = config.getboolean('gen_ctx_embedding')
        self.gen_ctx_file = config.get('gen_ctx_file')
        self.gen_ctx_dict_file = config.get('gen_ctx_dict_file')

        # label
        self.gen_label_embedding = config.getboolean('gen_label_embedding')
        self.gen_label_file = config.get('gen_label_file')
        self.gen_label_dict_file = config.get('gen_label_dict_file')

    def _get_train_config(self, config):
        """Read the configurations for training process.
        """
        self.rate = config.getfloat('rate', fallback=0.01)
        # self.rate = 2 ** self.rate
        self.max_iter = config.getint('max_iter', fallback=10)
        self.batch_size = config.getint('batch_size', fallback=100)
        self.embed_size = config.getint('embed_size')
        self.modeling = config.getint('modeling', fallback=2)

        self.V_reg = config.get('V_reg')
        self.W_reg = config.get('W_reg')
        regularizer = set(['OnBall', 'IntoBall', 'None'])
        if self.V_reg not in regularizer:
            raise Exception('Undefined regularizer !')
        if self.W_reg not in regularizer:
            raise Exception('Undefined regularizer !')

        self.weight = config.get('weight').strip()
        weights = set(['1', 'glove', 'frac', 'cbow'])
        if self.weight not in weights:
            raise Exception('Undefined weight function !')

        self.W_zero = config.getboolean('W_zero')
        self.V_zero = config.getboolean('V_zero')

        self.dropout_prob = config.getfloat('dropout_prob')
        self.ng_size = config.getint('ng_size')

    def _get_cv(self, config):
        self.cv_max_iter = config.getint('max_iter')
        self.cv_rates = config.get('rates').split()
        self.cv_rates = [float(v) for v in self.cv_rates]
