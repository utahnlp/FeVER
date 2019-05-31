# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-11-14 13:16:12
# Last modified: 2019-04-05 09:53:40

"""
Loading the training data.
"""

import multiprocessing as mp
import math
import os
import logging
import random

import torch
import numpy as np

import Logger
from FeVER.utils import _parse_line
import FeVER.utils as utils

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
Tensor = torch.Tensor
# Done = mp.Event()


def _processor(raw_queue, data_queue,
               label_feat, sampleTable, pid, args):
    packs = []
    while True:
        batch = raw_queue.get()
        if batch is None:
            raw_queue.put(None)
            data_queue.put(None)
            # Done.wait()
            break
        else:
            for item in batch:
                line = item[0]
                y, x = _parse_line(line)
                s = item[1].strip().split(' ')
                counts = [int(t) for t in s]

                s = item[2].strip().split()
                weights = [float(t) for t in s]
                packs += _preprocess(x, y, counts, weights, args)
                if len(packs) >= args.batch_size:
                    value = packs[:args.batch_size]
                    value = _collate_fn(value, label_feat, sampleTable, args)
                    data_queue.put(value)
                    packs = packs[args.batch_size:]


def _ng_sampler(ng_size, batch_size, sp_y,
                label_feat, sampleTable, args):
    """ Negative sampling.

    Args:
        ng_size: int, the size of each negative block.
        batch_size: int
        sp_y: list - The index of gloden label.
        label_feat: list - A list of feature index.
        sampleTable: sampling table.
    """
    reval = []
    sample_index = []
    assert batch_size == len(sp_y)
    for i in range(batch_size):
        tmp = []
        if args.modeling in utils.LOCAL:
            # Put the golden label feature
            tmp.append(label_feat[sp_y[i]])
            # Gather negative samples.
            samples = sampleTable.sample(ng_size-1)
            for j in samples:
                tmp.append(label_feat[j])
            sample_index.append(0)
        elif args.modeling in utils.GLOBAL:
            n = min(ng_size, len(sp_y[i]))
            # Put gloden label features
            sample_index.append(random.sample(range(len(sp_y[i])),
                                n))
            for j in sample_index[-1]:
                tmp.append(label_feat[sp_y[i][j]])

            # Gather negative samples.
            if len(tmp) < ng_size:
                samples = sampleTable.sample(ng_size-n)
                for j in samples:
                    tmp.append(label_feat[j])
        else:
            raise Exception('Wrong mode !')
        reval += tmp
    assert len(reval) == batch_size * ng_size
    return reval, sample_index


def _raw_reader(trainFile, countFile, weightFile, raw_queue, batch_size):
    """
    """
    batch = []
    for item, count, weight in zip(trainFile, countFile, weightFile):
        batch.append((item, count, weight))
        if len(batch) == batch_size:
            raw_queue.put(batch)
            batch = []
    if len(batch) != 0:
        raw_queue.put(batch)
        batch = []
    raw_queue.put(None)


def _preprocess(sp_x, sp_y, counts, weights, args):
    """
    Reorganize the dataset according to normalization.
    If it is gloabl, it remains the same.
    If it is local,  the data will be flattened.
    """
    if args.modeling in utils.LOCAL:
        reval = []
        assert len(sp_y) == len(counts)
        assert len(sp_y) == len(weights)
        for y, c, w in zip(sp_y, counts, weights):
            reval.append((sp_x, y, c, w))
        return reval
    else:
        return [(sp_x, sp_y, counts, weights)]


def _collate_fn(data, label_feat, sampleTable, args):
    prob = args.dropout_prob
    x = [t[0] for t in data]
    # x = data[0]

    sx, xoffsets = _tensor(x, prob)

    y = [t[1] for t in data]

    # Perform negative sampling
    sample, sample_index = _ng_sampler(args.ng_size, len(data), y,
                                       label_feat, sampleTable, args)

    # Bagging
    sy, yoffsets = _tensor(sample, 1.0)

    counts = [t[2] for t in data]
    weights = [t[3] for t in data]
    # y = data[1]
    # counts = data[2]
    # weights = data[3]
    if args.modeling in utils.LOCAL:
        # Because of negative sampling, golden id
        # is always 0.
        y = LongTensor([0]*len(y))
        counts = FloatTensor(counts)
        weights = FloatTensor(weights)

    if args.modeling in utils.GLOBAL:
        # Because of negative sampling, golden id
        # is always range(len(w)).
        y = [list(range(min(len(w), args.ng_size))) for w in y]
        tcounts = []
        tweights = []
        for i, index in enumerate(sample_index):
            tcounts.append([counts[i][j] for j in index])
            tweights.append([weights[i][j] for j in index])
        counts = tcounts
        weights = tweights
    return ((sx, xoffsets), y, counts, weights, (sy, yoffsets))


def _tensor(data: list, prob: float):
    """This function applies following things:
    1. Apply dropout
    2. bagging
    3. Transfer to tensor
    """
    x = utils.dropout(data, prob)
    s, offsets = utils.bag(x)
    s = torch.from_numpy(s).long()
    offsets = torch.from_numpy(offsets).long()
    return s, offsets


class _DataSet:
    def __init__(self, args, label_feat, sampleTable):
        """Iterator of reading large file.
        """
        # Done.clear()
        self.data_queue = mp.Manager().Queue(20)
        # self.data_queue = mp.Queue(20)
        self.raw_queue = mp.Queue(20)
        self.args = args
        self.label_feat = label_feat
        self.sampleTable = sampleTable
        with open(args.train_file, encoding='utf8') as trainFile:
            with open(args.counts_file, encoding='utf8') as countFile:
                with open(args.weight_file, encoding='utf8') as weightFile:
                    nums = next(trainFile)
                    nums = nums.split()
                    self.length = math.ceil(int(nums[0]) / args.batch_size)
                    self.counter = 0

                    self.workers = [mp.Process(
                               target=_processor,
                               args=(self.raw_queue, self.data_queue,
                                     label_feat, sampleTable, i, args))
                                    for i in range(args.num_workers)]
                    self.reader = mp.Process(
                                target=_raw_reader,
                                args=(trainFile, countFile, weightFile,
                                      self.raw_queue, args.batch_size))

                    self.reader.daemon = True
                    self.reader.start()
                    for w in self.workers:
                        w.daemon = True
                        w.start()

    def __next__(self):
        while self.counter != self.args.num_workers:
            value = self.data_queue.get()
            if value is not None:
                # value = _collate_fn(value, self.label_feat,
                #                     self.sampleTable, self.args)
                return value
            else:
                self.counter += 1
        # Done.set()
        self.reader.join()
        for worker in self.workers:
            worker.join()
        raise StopIteration

    def __len__(self):
        return self.length


class XMLDataset:
    def __init__(self, args):
        self._args = args
        logger = logging.getLogger(Logger.project_name)

        train_path = args.train_file
        counts_path = args.counts_file
        label_path = args.label_file

        # Rreading meta info
        s = 'Reading meta information...'
        logger.info(s)
        nums = utils.read_meta(train_path)
        self._input_feat_num = nums.feat_num

        # Reading label features

        s = 'Reading label features...'
        logger.info(s)
        nums, label_feat, _ = utils.read_binary(label_path, args)
        self._label_feat_num = nums.feat_num
        self._label_num = nums.ins_num
        self._label_feat = label_feat

        self._local = utils.LOCAL
        self._glob = utils.GLOBAL

        s = 'Reading labels ...'
        logger.info(s)
        y = utils.read_labels(train_path, args)

        # Reading counts
        s = 'Reading counts ...'
        logger.info(s)
        counts = utils.read_counts(counts_path)

        s = 'Gather the frequency of labels'
        logger.info(s)
        # Gather the frequency of labels
        table = self._gather_frequency(y, counts)

        s = 'Construct unigram table...'
        self.sampleTable = UnigramTable(table)

        if not os.path.exists(args.weight_file):
            total_label_count = [count for count in table.values()]
            self.total_label_count = sum(total_label_count)
            s = 'Calculating the weights...'
            logger.info(s)
            # Calculate the weights
            weights = self._cal_weights(y, table)

            s = 'Writing the weights...'
            logger.info(s)
            utils.write_weights(args.weight_file, weights)
            del table
            del y
            del weights

        # s = 'Bagging the label features...'
        # logger.info(s)
        # self._label_feat = self._bagging(self._label_feat)

    def __iter__(self):
        return _DataSet(self._args, self._label_feat, self.sampleTable)

    def _bagging(self, features: list):
        """Bagging the fearures.

        Args:
            fearures: A list of np.int32
        """
        label_feat, offsets = utils.bag(features)
        label_feat = torch.from_numpy(label_feat).long()
        offsets = torch.from_numpy(offsets).long()
        if self._args.cuda is True:
            offsets = offsets.cuda(utils.CUDA0)
            label_feat = label_feat.cuda(utils.CUDA0)
        return (label_feat, offsets)

    @property
    def input_feat_num(self):
        return self._input_feat_num

    @property
    def label_feat_num(self):
        return self._label_feat_num

    @property
    def label_num(self):
        return self._label_num

    @property
    def label_feat(self):
        return self._label_feat

    def _cal_weights(self, sp_y: list, table: dict)->list:
        """ Calculate the weights for each label.
        """
        weights = []
        for Y in sp_y:
            tmp = [self._weight_func(table[i]) for i in Y]
            weights.append(tmp)
        return weights

    def _weight_func(self, freq: float)->float:
        """Calculate the weight based on given frequence.
        """
        args = self._args
        if args.weight == '1':
            return 1
        elif args.weight == 'frac':
            return freq ** (self._args.alpha)
        elif args.weight == 'cbow':
            t = 0.001 * self.total_label_count
            t = t/freq
            x = (t)**(0.5) + t
            return min(x, 1)
        else:
            raise Exception('Undefined weight function !')

    def _gather_frequency(self, y: list, counts: list)-> dict:
        """Gather a frequency table for given dataset.

        sp_y: list(list(int))
        counts: list(list(int))
        """
        table = dict()
        assert len(y) == len(counts)
        for Y, C in zip(y, counts):
            assert len(Y) == len(C)
            for i, c in zip(Y, C):
                table[i] = table.get(i, 0) + c
        return table


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution
    used to draw negative samples.
    """
    def __init__(self, vocab: dict):
        logger = logging.getLogger(Logger.project_name)
        power = 0.75
        # Normalizing constant
        norm = sum([math.pow(t, power) for t in vocab.values()])
        # Length of the unigram table
        table_size = int(1e8)
        table = np.zeros(table_size, dtype=np.uint32)

        s = 'Sorting the vocabulary...'
        logger.info(s)
        tmp = [(key, value) for key, value in vocab.items()]
        tmp.sort(key=lambda x: x[1], reverse=True)

        s = 'Filling unigram table'
        logger.info(s)
        # Cumulative probability
        p = 0
        i = 0
        for key, count in tmp:
            p += float(math.pow(count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = key
                i += 1
        self.table = table

    def sample(self, ng_size):
        indices = np.random.randint(low=0, high=len(self.table), size=ng_size)
        return [self.table[i] for i in indices]
