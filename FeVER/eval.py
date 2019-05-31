# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-10-30 08:39:01
# Last modified: 2018-10-30 12:45:20

"""
Evaluate XML.
"""

import torch


FloatTensor = torch.FloatTensor


class Eval:
    def __init__(self):
        pass

    def eval(self, scores: FloatTensor, dev_labels: list, k: int):
        """Evaluate the scores using top k.
        """
        reval = []
        scores, indexs = torch.topk(scores, k, 1)
        predicts = indexs.tolist()
        assert len(predicts) == len(dev_labels)
        for pred, gold in zip(predicts, dev_labels):
            common = set(gold) & set(pred)
            reval.append(len(common)/k)
        return sum(reval) / len(reval)
