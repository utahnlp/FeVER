# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-04-19 09:07:41
# Last modified: 2018-09-26 13:54:53

"""
Test.
"""
import time

import torch

M = 6500


# def non_batch():
# 
#     W = torch.nn.Embedding(5, 300).cuda()
# 
#     indexs = [torch.LongTensor([1, 2, 3]), torch.LongTensor([0, 4])]*M
#     indexs = [t.cuda() for t in indexs]
# 
#     s = time.perf_counter()
#     reval = []
#     for vec in indexs:
#         p = W(vec)
#         q = torch.sum(p, 0)
#         reval.append(q)
#     e = time.perf_counter()
# 
#     vec = torch.stack(reval, 0)
#     print(e-s)
# 
# 
def batch():
    W = torch.nn.Embedding(5, 300).cuda()
    indexs = torch.LongTensor([[1, 2, 3], [0, 2, 4]]*M).cuda()
    s = time.perf_counter()
    vec = W(indexs)
    e = time.perf_counter()
    print(e-s)
    vec = torch.sum(vec, 1)
# 
# 
# def sample():
#     W = torch.nn.Embedding(5, 300).cuda()
#     indexs = [torch.LongTensor([1, 2, 3]), torch.LongTensor([0, 4])] * M
#     indexs = [t.cuda() for t in indexs]
# 
#     # 1
#     n = min([len(t) for t in indexs])
#     m = max([len(t) for t in indexs])
#     # 2
#     probs = torch.zeros((len(indexs), m)).cuda()
# 
#     # 3
#     for i, j in enumerate(indexs):
#         probs[i][:len(indexs[i])] = 1
# 
#     # 4
#     x = torch.multinomial(probs, n, replacement=False).cuda()
# 
#     s = time.perf_counter()
#     myindexs = [indexs[i][j] for i, j in enumerate(x)]
#     e = time.perf_counter()
#     print(e-s)
#     exit()
# 
#     myindexs = torch.stack(myindexs, 0)
# 
#     vec = W(myindexs)
#     vec = torch.sum(vec, 1)
# 
# 

def concate():
    W = torch.nn.EmbeddingBag(5, 300, mode='sum').cuda()
    indexs = [torch.LongTensor([1, 2, 3]), torch.LongTensor([0, 4])] * M
    indexs = [t.cuda() for t in indexs]

    s = time.perf_counter()
    offsets = [0]
    for feat in indexs[:-1]:
        offsets.append(offsets[-1]+len(feat))
    offsets = torch.LongTensor(offsets).cuda()
    myindexs = torch.cat(indexs)

    vecs = W(myindexs, offsets)
    e = time.perf_counter()
    print(e-s)


if __name__ == '__main__':
    batch()
#     # non_batch()
#     # sample()
    concate()
