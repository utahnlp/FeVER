# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-04-16 13:58:29
# Last modified: 2019-04-05 09:54:43

"""
A collection of different loss functions.

For now, we have following loss functions:
    1. Local ranking loss
    2. Local hinge loss
    3. Local log loss
    4. global hinge loss
"""

import torch

import FeVER.utils as utils


class LossFuntion:
    def __init__(self, args):
        self.LogLoss = torch.nn.CrossEntropyLoss(reduction='none')
        self.args = args

        if args.modeling in utils.GLOBAL:
            M1 = torch.FloatTensor(args.batch_size, args.ng_size).fill_(1)
            M2 = torch.FloatTensor(args.batch_size, args.ng_size).fill_(0)
            M3 = torch.FloatTensor(args.batch_size, args.ng_size).fill_(1)

            if args.cuda:
                M1 = M1.cuda(utils.CUDA1)
                M2 = M2.cuda(utils.CUDA1)
                M3 = M3.cuda(utils.CUDA1)
            self.M1 = M1
            self.M2 = M2
            self.M3 = M3

    def local_hinge(self, scores, batch):
        """
        Args:
            scores: Tensor - The score table of the model
            batch[0]: Tensor - Input features
            batch[1]: Tensor - A list of labels
            batch[2]: Tensor - Frequency of each label in batch[1]
            batch[3]: Tensor - Weight of each lable in batch[1]
        """
        ys = batch[1]
        counts = batch[2]
        weights = batch[3]

        assert len(scores) == len(ys)

        T = scores
        gold_scores = T[:, :1]
        assert len(gold_scores) == len(ys)
        assert len(T) == len(ys)
        # Deal with special case
        # when scores are all zero.
        if len(torch.nonzero(T)) == 0:
            T, indexs = torch.max(T[:, 1:], 1)
            delta = (indexs+1 != ys).float()
        else:
            T, indexs = torch.max(T, 1)
            delta = (indexs != ys).float()

        P = delta + T - gold_scores.view(-1)

        assert len(P) == len(counts)
        assert len(P) == len(weights)
        loss = P*counts*weights
        assert len(loss) == len(scores)

        loss = torch.sum(loss)

        return loss

    def global_hinge(self, scores, batch):
        Ys = batch[1]
        counts = batch[2]
        weights = batch[3]
        T = scores

        self.M1.fill_(1)
        self.M2.fill_(1)

        a = []
        for i, Y in enumerate(Ys):
            a += ([i]*len(Y))
        try:
            b = torch.cat(Ys)
        except RuntimeError:
            print('I am here !')
            print(Ys)
        c = [p*q for p, q in zip(counts, weights)]
        c = torch.cat(c)
        self.M1[a, b.data] = -1
        self.M2[a, b.data] = c.data

        TT = self.M2 + T*self.M1
        mask = (TT > 0).float()

        loss = torch.sum(mask*TT)
        return loss

    def local_log(self, scores, batch):
        """Local log loss
        """
        ys = batch[1].view(-1)
        counts = batch[2].view(-1)
        weights = batch[3].view(-1)

        T = scores

        loss = self.LogLoss(T, ys)
        assert len(loss) == len(ys)
        loss = loss*counts*weights
        assert len(loss) == len(scores)

        loss = torch.sum(loss)

        return loss

    def global_log(self, scores, batch):
        """Global log loss.
        """
        Ys = batch[1]
        counts = batch[2]
        weights = batch[3]
        T = scores

        self.M1.fill_(1)
        self.M2.fill_(0)
        a = []
        for i, Y in enumerate(Ys):
            a += ([i]*len(Y))
        try:
            b = torch.cat(Ys)
        except RuntimeError:
            print('I am here !')
            print(Ys)

        c = [p*q for p, q in zip(counts, weights)]
        c = torch.cat(c)
        self.M1[a, b] = c.data
        M1 = self.M1
        TT = T * M1
        assert len(TT) == self.args.batch_size

        self.M2[a, b] = 1
        A = self.M2*TT
        A = torch.sum(A, 1)
        assert len(A) == self.args.batch_size

        M1 = TT > 15
        M2 = TT < -15

        self.M3.fill_(1)
        S = torch.where(TT > 15, self.M3, TT)
        S = torch.where(S < -15, self.M3, S)

        S = torch.log(torch.exp(S) + 1)
        S[M1] = TT[M1]
        S[M2] = 0
        B = torch.sum(S, 1)
        assert len(B) == self.args.batch_size

        loss = -1 * A + B
        loss = torch.sum(loss)

        return loss


# def local_ranking(self, batch, model):
#     """Compute the local ranking loss.
#     """
#     Xs = batch[0]
#     ys = batch[1]
#
#     # For the scoring
#     T = model.Xscore(Xs)
#
#     self.M1.fill_(0)
#     for i, y in enumerate(ys):
#         self.M1[:, i][y] = 1
#
#     M1 = Variable(self.M1, requires_grad=False)
#     # M2 = Variable(self.M2, requires_grad=False)
#
#     # Margin loss
#     # indexs = ys
#     min_scores = torch.transpose(T, 0, 1).masked_select(
#             torch.transpose(M1, 0, 1))
#
#     # For debug
#     # for i, s in enumerate(min_scores.data):
#     #     assert T.data[:, i][indexs[i]] == s
#
#     # P = (T + 1 - min_scores) * M2
#     P = (T + 1 - min_scores)
#     N = P.gt(0)
#
#     N = N.float()
#     rank = torch.sum(N, 0)
#     loss = torch.sum(P*N, 0)
#
#     L = [model.L(int(v)) for v in rank.data]
#     L = Variable(torch.FloatTensor(L).type(self.dtype),
#                  requires_grad=False)
#
#     mask = (rank != 0).detach()
#     L = L[mask]
#     rank = rank[mask]
#     loss = loss[mask]
#
#     loss = L * (1/rank) * loss
#     loss = torch.sum(loss)
#     return loss
