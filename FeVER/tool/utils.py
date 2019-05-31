# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-11-28 13:35:02
# Last modified: 2018-11-28 21:09:24

"""
Common used functions
"""


def write_freq(path, freqs):
    with open(path, 'w', encoding='utf8') as f:
        for item in freqs:
            s = [str(t) for t in item]
            s = ' '.join(s)
            f.write(s+'\n')


def write_vocab(path, vocab):
    with open(path, 'w', encoding='utf8') as f:
        for key, value in vocab.items():
            s = [str(key), str(value)]
            s = '$:$'.join(s)
            f.write(s+'\n')


def read_vocab(path):
    reval = dict()
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split('$:$')
            word = s[0]
            ID = int(s[1])
            reval[word] = ID
    return reval
