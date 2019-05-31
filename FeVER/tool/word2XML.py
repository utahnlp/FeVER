# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-11-28 10:53:55
# Last modified: 2018-11-28 20:44:11

"""
Transform the text corpus into XML format.

    - Generating the vocabulary based on the corpus.
    - Gathering all the (context, label) pairs.
"""

from collections import Counter
import logging

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

WIN_SIZE = 5


def read_corpus(path):
    """Read the corpus.
    """
    logger.info('Reading text...')
    reval = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            reval += s
    logger.info('Finish reading text.')
    return reval


def make_vocab(data, threshold=5):
    """ Generate the dictionary by the data.
    """
    logger.info('Counting words...')
    counter = Counter(data)
    common = counter.most_common()
    common = [(word, freq) for word, freq in common if freq >= threshold]
    s = 'Total words: {a}'.format(a=str(len(data)))
    logger.info(s)

    words = [freq for word, freq in common]
    n = sum(words)

    s = 'Training words: {a}'.format(a=str(n))
    logger.info(s)

    s = 'Unique words: {a}'.format(a=str(len(common)))
    logger.info(s)

    vocab = dict()

    s = 'Building vocabulary...'
    logger.info(s)
    for i, (word, _) in enumerate(common):
        vocab[word] = i
    return vocab


def extract_pairs(data):
    """
    Extract (context, target) from the given corpus.

    Args:
        data: np.array()
    """
    reval = []
    logger.info('Extracting (context, target) pairs...')

    n = len(data)
    for index in tqdm(range(WIN_SIZE, n-WIN_SIZE)):
        pair = data[index-WIN_SIZE:index+WIN_SIZE+1]
        assert len(pair) == 2 * WIN_SIZE + 1
        reval.append(tuple(pair))
    s = 'Total pairs: {a}'.format(a=str(len(reval)))
    logger.info(s)

    logger.info('Counting the frequency of (context, target) pair...')
    counter = Counter(reval)
    common = counter.most_common()
    return common


def gather(common):
    """ Group all the target words with same context.
    """
    data = dict()
    logger.info('Gathering all the instances with same context...')
    for pair, freq in tqdm(common):
        target = pair[WIN_SIZE]
        context = pair[:WIN_SIZE] + pair[WIN_SIZE+1:]
        assert len(context) == 2 * WIN_SIZE
        if context not in data:
            data[context] = []
        data[context].append((target, freq))

    contexts = []
    targets = []
    freqs = []
    logger.info('Sequence the mutli-label exampels...')
    for ctx, values in tqdm(data.items()):
        contexts.append(ctx)
        target = tuple([t[0] for t in values])
        freq = tuple([t[1] for t in values])
        targets.append(target)
        freqs.append(freq)

    s = 'Total multi-label example: {a}'
    s = s.format(a=str(len(contexts)))
    logger.info(s)
    return contexts, targets, freqs


def generate_training_examples(corpus_path):
    raw_data = read_corpus(corpus_path)
    vocab = make_vocab(raw_data)

    logger.info('Encoding the raw text...')
    train_data = [vocab[word] for word in raw_data if word in vocab]
    del raw_data
    train_data = np.array(train_data, dtype=np.uint32)
    common = extract_pairs(train_data)
    data = gather(common)

    return data, vocab
