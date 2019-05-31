# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2017-03-03 13:51:52
# Last modified: 2018-12-08 13:54:12

"""
IO manager for wsabie system.
"""

from collections import namedtuple
import multiprocessing
import numpy as np

import torch


Nums = namedtuple('Nums', ['ins_num', 'feat_num', 'label_num'])
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
Tensor = torch.Tensor
LOCAL = set([1, 3])
GLOBAL = set([2, 4])

if torch.cuda.device_count() >= 2:
    CUDA0 = torch.device('cuda:0')
    CUDA1 = torch.device('cuda:1')
else:
    CUDA0 = torch.device('cuda:0')
    CUDA1 = torch.device('cuda:0')


def read_binary(path: str, args)->(Nums, list, list):
    """Read the input data.

    All the features need to be binary.
    This is especially used for learning word embedding.

    Args:
        path: The path of input file.

    Returns:
        nums:
        x: A list of feature id.
        y: A list of label.
    """
    y = []
    x = []
    with open(path, encoding='utf8') as f:
        line = next(f)
        s = line.strip().split()
        ins_num = int(s[0])
        feat_num = int(s[1])
        label_num = int(s[2])
        nums = Nums(ins_num, feat_num, label_num)
        with multiprocessing.Pool(args.num_workers) as pool:
            results = pool.imap(_parse_line, f, 1000)
            pool.close()
            pool.join()
            x = []
            y = []
            # results is a iterator
            for item in results:
                x.append(item[1])
                y.append(item[0])

    return nums, x, y


def _parse_line(line: str) -> (list, list):
    """Parse the line into Instance.

    Input:
        1,2 1:4.5 2:3

    Returns:
        - list(int): A list of label.
        - list(tuple(int, float)): A dictionary from feature
                            id to feature value
    """
    s = line.strip().split()
    # Deal with case that there are no labels at all
    if ':' in s[0]:
        labels = []
        f = s
    else:
        labels = s[0].split(',')
        labels = [int(v) for v in labels]
        labels = sorted(labels)
        f = s[1:]
    x = []
    for item in f:
        p = item.split(':')
        x.append(int(p[0]))
    return labels, np.int32(x)


def _parse_label(line: str) -> list:
    """Only parse the lables.
    """
    s = line.strip().split()
    if ':' in s[0]:
        labels = []
    else:
        labels = s[0].split(',')
        labels = [int(v) for v in labels]
        labels = sorted(labels)
    return labels


def read_counts(path: str)->list:
    """Read the counts for each (context, word) pair.
    """
    counts = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split(' ')
            s = [int(x) for x in s]
            counts.append(s)
    return counts


def read_labels(path: str, args)->list:
    """Read the label ids from the training file.
    """
    with open(path, encoding='utf8') as f:
        next(f)
        with multiprocessing.Pool(args.num_workers) as pool:
            results = pool.imap(_parse_label, f, 1000)
            pool.close()
            pool.join()
            y = []
            # results is a iterator
            for item in results:
                y.append(item)
    return y


def load_mapping(path: str)->dict:
    reval = dict()
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split('$:$')
            word = s[0]
            ID = int(s[1])
            reval[ID] = word
    return reval


def load_feat_mapping(path: str) -> dict:
    reval = dict()
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split('$:$')
            word = s[0][1:-1]
            ID = int(s[1])
            reval[ID] = word
    return reval


def write_embed(path: str, embed: list)->None:
    with open(path, 'w', encoding='utf8') as f:
        n = len(embed)
        dim = len(embed[0][-1])
        s = [str(n), str(dim)]
        s = ' '.join(s)
        f.write(s+'\n')
        for word, v in embed:
            v = [format(i, '0.4f') for i in v]
            s = [word] + v
            s = ' '.join(s)
            f.write(s+'\n')


def bag(data: list)->(list, list):
    """ Pack the fearture index into a huge tensor.
    """
    offsets = [0]
    for item in data[:-1]:
        offsets.append(offsets[-1]+len(item))
    # offsets = LongTensor(offsets)
    return np.concatenate(data), np.int32(offsets)


def dropout(x: list, prob: float)->list:
    """Apply dropout on the input list.

    Discard some elements of each context with probability 1-p.
    """
    reval = []
    for item in x:
        if item.shape[0] <= 2:
            reval.append(item)
            continue
        # Make sure there at least one feature is activated.
        while True:
            mask = np.random.rand(item.shape[0]) < prob
            if len(item[mask]) > 1:
                break
        reval.append(item[mask])

    assert len(reval) == len(x)
    return reval


def write_cv(results: tuple, path: str):
    with open(path, 'w') as f:
        for rate, top5 in results:
            s = 'rate={a}, top5={b}\n'.format(a=str(rate),
                                              b=str(top5))
            f.write(s)
        results = sorted(results, key=lambda x: x[1])
        s = 'Best setting:\n'
        f.write(s)
        rate, top5 = results[-1][0], results[-1][1]
        s = 'rate={a}, top5={b}\n'.format(a=str(rate),
                                          b=str(top5))
        f.write(s)


def batch2GPU(batch: tuple, args):
    """Move the whole batch to the gpu.
    """
    x = batch[0]
    y = batch[1]
    counts = batch[2]
    weights = batch[3]
    sx = x[0]
    xoffsets = x[1]

    sy = batch[-1][0]
    yoffsets = batch[-1][1]

    if args.modeling in GLOBAL:
        y = [LongTensor(t) for t in y]
        counts = [FloatTensor(t) for t in counts]
        weights = [FloatTensor(t) for t in weights]

    if args.cuda is True:
        sx = sx.cuda(CUDA0)
        xoffsets = xoffsets.cuda(CUDA0)
        sy = sy.cuda(CUDA0)
        yoffsets = yoffsets.cuda(CUDA0)
        if args.modeling in LOCAL:
            y = y.cuda(CUDA1)
            counts = counts.cuda(CUDA1)
            weights = weights.cuda(CUDA1)
        elif args.modeling in GLOBAL:
            y = [w.cuda(CUDA1) for w in y]
            counts = [w.cuda(CUDA1) for w in counts]
            weights = [w.cuda(CUDA1) for w in weights]
        else:
            raise Exception('Undefined modeling !')
    return ((sx, xoffsets), y, counts, weights, (sy, yoffsets))


def read_meta(path):
    with open(path, encoding='utf8') as f:
        line = next(f)
        s = line.strip().split()
        ins_num = int(s[0])
        feat_num = int(s[1])
        label_num = int(s[2])
        nums = Nums(ins_num, feat_num, label_num)
    return nums


def write_weights(path, weights):
    with open(path, 'w', encoding='utf8') as f:
        for example in weights:
            s = [str(t) for t in example]
            s = ' '.join(s)
            f.write(s+'\n')
