# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-10-04 11:18:48
# Last modified: 2018-10-04 14:56:27

"""
Test code for multiprocessing reading large file.
"""
import multiprocessing
import time
from tqdm import tqdm

PATH = '../../embedding_data/word_noindex/example.txt'
NUM = 22616247


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
        x.append((int(p[0]), float(p[1])))

    indexs = [int(s[0]) for s in x]
    return labels, indexs


def parallel():
    pool = multiprocessing.Pool(15)
    f = open(PATH, encoding='utf8')
    next(f)
    s = time.perf_counter()
    results = pool.imap_unordered(_parse_line, f, 1000)
    results = list(tqdm(results, total=NUM))
    pool.close()
    print(len(results))
    print(results[-1])
    e = time.perf_counter()
    print(e-s)


def single():
    results = []
    with open(PATH, encoding='utf8') as f:
        next(f)
        s = time.perf_counter()
        for i, line in enumerate(f):
            results.append(_parse_line(line))
            if i % 1000 == 0:
                ss = ' '*100
                print(ss, end='\r')
                ss = 'Progress: {a}%'
                ss = ss.format(a=str(i/22616246*100))
                print(ss, end='\r')
    print()
    print(len(results))
    e = time.perf_counter()
    print(e-s)


if __name__ == '__main__':
    parallel()
    # single()
