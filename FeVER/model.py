# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2017-03-07 11:29:06
# Last modified: 2019-04-05 09:55:27

"""
The model class for WSABIE model.
"""

import logging

import torch
import torch.nn as nn

from FeVER import Logger
import FeVER.utils as utils

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
EmbeddingBag = nn.EmbeddingBag


class Model(nn.Module):
    def __init__(self, parameters):
        """Construct a new wsabie model.

        Args:
            parameters: dict - The parameters of this model.
        """
        super(Model, self).__init__()
        self._args = parameters
        self.W, self.V = self._init_para()

    def forward(self, x: LongTensor, y: LongTensor,
                xoffsets: LongTensor, yoffsets: LongTensor)->FloatTensor:
        """Score function.

        Args:
            x:  Feature vector of input example in shape of 1D.
            This tensor is a concatenation of multiple sequence.

            y:  Feature vector of labels in shape of 1D.
            This tensor is a concatenation of multiple sequence.

            xoffsets: Contains the starting positions of each
            sequence in x.

            yoffsets: Contains the starting positions of each
            sequence in y.

        Return:
            A Tensor of score, batch_size x label_size.
        """
        args = self._args
        phi_x = self.V(x, xoffsets)

        phi_y = self.W(y, yoffsets)

        if args.cuda is True:
            phi_y = phi_y.cuda(utils.CUDA1)
            phi_x = phi_x.cuda(utils.CUDA1)

        assert len(phi_y) == len(yoffsets)
        assert len(phi_x) == len(xoffsets)
        phi_y = phi_y.reshape(-1, args.ng_size, args.embed_size)
        phi_x = phi_x.reshape(-1, args.embed_size, 1)

        reval = phi_y.matmul(phi_x)
        reval = reval.reshape(-1, args.ng_size)

        return reval

    def label_embed_func(self, y: LongTensor, yoffsets: LongTensor,
                         mapping: dict, index: list)-> list((str, list)):
        """
        Args:
            y:  Feature vector of labels in shape of 1D.
            This tensor is a concatenation of multiple sequence.

            yoffsets: Contains the starting positions of each
            sequence in y.
        """
        return self._embedding(y, yoffsets, mapping, index, self.W)

    def context_embed_func(self, x: LongTensor, xoffsets: LongTensor,
                           mapping: dict, index: list) -> list((str, list)):
        return self._embedding(x, xoffsets, mapping, index, self.V)

    ########################################################
    # Private methods
    ########################################################
    def _init_para(self)->(EmbeddingBag, EmbeddingBag):
        """Initiaize the parameters.

        Returns:
            W: np.array - The weight matrix for label.
            V: np.array - The weight matrix for features.
        """
        logger = logging.getLogger(Logger.project_name)
        args = self._args

        if args.W_reg == 'IntoBall':
            W = EmbeddingBag(args.label_feat_num,
                             args.embed_size,
                             mode='mean', max_norm=1)
        else:
            W = EmbeddingBag(args.label_feat_num,
                             args.embed_size,
                             mode='mean')
        if args.V_reg == 'IntoBall':
            V = EmbeddingBag(args.input_feat_num,
                             args.embed_size,
                             mode='mean', max_norm=1)
        else:
            V = EmbeddingBag(args.input_feat_num,
                             args.embed_size,
                             mode='mean')

        if args.W_zero is True:
            W.weight.data.zero_()
        else:
            W.weight.data.uniform_(-0.5/args.embed_size, 0.5/args.embed_size)
        if args.V_zero is True:
            V.weight.data.zero_()
        else:
            V.weight.data.uniform_(-0.5/args.embed_size, 0.5/args.embed_size)

        # For debug
        # index = torch.LongTensor([1190])
        # print(V(index))

        # index = torch.LongTensor([119])
        # print(W(index))
        # exit()

        if args.cuda is True:
            W = W.cuda(utils.CUDA0)
            V = V.cuda(utils.CUDA0)
        logger.info('Finish initializeing model parameters...')
        return W, V

    def _embedding(self, feats: LongTensor, offsets: LongTensor,
                   mapping: dict, index: list,
                   embedding: EmbeddingBag)-> list((str, list)):
        phi = embedding(feats, offsets)
        reval = []
        assert len(phi) == len(offsets)
        for i, embed in enumerate(phi):
            reval.append((mapping[index[i]], embed.tolist()))
        return reval
