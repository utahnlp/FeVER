# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-11-30 10:12:29
# Last modified: 2019-03-02 13:22:57

"""
Select the embeddings based on different vocabulary.
"""

from pathlib import Path
import os

import numpy as np

from gel.tool import vocab


def read_embeddings(path):
    reval = dict()
    with open(path, encoding='utf8') as f:
        s = next(f).strip().split()
        embed_size = int(s[1])
        for line in f:
            s = line.strip().split()
            word = s[0]
            vec = np.array([float(t) for t in s[1:]])
            reval[word] = vec
    return reval, embed_size


def read_embed_vocab(path):
    reval = set()
    with open(path, encoding='utf8') as f:
        next(f)
        for line in f:
            s = line.strip().split()
            word = s[0]
            reval.add(word)
    return reval


def three_percent():
    print('Reading three percent training vocabulary...')
    path = '/home/flyaway/scr/word2vec/embeddings/three_corpus_cbow.txt'
    words = read_embed_vocab(path)
    return words


def ten_percent():
    print('Reading ten percent training vocabulary...')
    path = '/home/flyaway/scr/word2vec/embeddings/ten_corpus_cbow.txt'
    words = read_embed_vocab(path)
    return words


def hundred_percent():
    print('Reading one hundred percent training vocabulary...')
    path = '/home/flyaway/scr/word2vec/embeddings/all_corpus_cbow.txt'
    words = read_embed_vocab(path)
    return words


def hundred_intrinsic():
    words = hundred_percent()
    print('Reading intrinsic datasets...')
    # words |= vocab.BattigSet()
    # words |= vocab.MENSet()
    # words |= vocab.SimLexSet()
    words |= vocab.GoogleSet()
    words |= vocab.MSRSet()
    return words


def extrinsic():
    print('Reading extrinsic datasets...')
    # words = vocab.YahooSet()
    words = vocab.NewsSet()
    return words


def hundred_typo_rare():
    print('Reading typo and rare datasets...')
    words = hundred_percent()
    words |= vocab.GoogleSet(typo=True)
    words |= vocab.MSRSet(typo=True)
    # words |= vocab.RWSet()
    return words


def write_embeddings(embeds, embed_size, path):
    n = len(embeds)
    with open(path, 'w', encoding='utf8') as f:
        s = [str(n), str(embed_size)]
        s = ' '.join(s)
        f.write(s+'\n')
        for word, vec in embeds.items():
            s = vec.tolist()
            s = [format(i, '0.4f') for i in s]
            s = [word] + s
            s = ' '.join(s)
            f.write(s+'\n')


def select_embeds(dest_path, vocab, source_embeds, embed_size):
    embeds = {w: source_embeds[w] for w in source_embeds.keys() if w in vocab}
    write_embeddings(embeds, embed_size, dest_path)


def selects(source_embed_path, dest_path, tag, vocabs):
    embeds, embed_size = read_embeddings(source_embed_path)
    """Select the embeddings based on different vocabulary.
    """

    # path = dest_path / (tag + '_three_embeds.txt')
    # print('Writing three percent vocabulary...')
    # select_embeds(
    #         path, vocabs['three_percent'], embeds, embed_size)

    path = dest_path / (tag + '_ten_embeds.txt')
    print('Writing ten percent vocabulary...')
    select_embeds(
            path, vocabs['ten_percent'], embeds, embed_size)

    path = dest_path / (tag + '_hundred_embeds.txt')
    print('Writing one hundred percent vocabulary...')
    select_embeds(
            path, vocabs['hundred_percent'], embeds, embed_size)

    path = dest_path / (tag + '_hundred_intrinsic.txt')
    print('Writing one hundred percent and intrinsic vocabulary...')
    select_embeds(
            path, vocabs['hundred_intrinsic'], embeds, embed_size)

    path = dest_path / (tag + '_extrinsic.txt')
    print('Writing extrinsic vocabulary...')
    select_embeds(
            path, vocabs['extrinsic'], embeds, embed_size)

    path = dest_path / (tag + '_hundred_typo_rare.txt')
    print('Writing one hundred percent, typo and rare words vocabulary...')
    select_embeds(
            path, vocabs['hundred_typo_rare'], embeds, embed_size)


def read_vocabs():
    reval = dict()
    # reval['three_percent'] = three_percent()
    reval['ten_percent'] = ten_percent()
    reval['hundred_percent'] = hundred_percent()
    reval['hundred_intrinsic'] = hundred_intrinsic()
    reval['extrinsic'] = extrinsic()
    reval['hundred_typo_rare'] = hundred_typo_rare()
    return reval


def main(path):
    path = Path(path)
    dest_path = path / 'embeddings'
    if not dest_path.exists():
        os.mkdir(dest_path)

    print('Reading different vocabularies...')
    vocabs = read_vocabs()

    context_embed_path = path / 'context_embedding.txt'
    tag = 'context'
    selects(context_embed_path, dest_path, tag, vocabs)

    label_embed_path = path / 'label_embedding.txt'
    tag = 'label'
    selects(label_embed_path, dest_path, tag, vocabs)


def write_vocab(path, vocab):
    print(path)
    with open(path, 'w', encoding='utf8') as f:
        for word in vocab:
            f.write(str(word)+'\n')


if __name__ == '__main__':
    # path = '../../Experiments/2707'
    # main(path)

    path = '~/scr/word2vec/vocabs/'
    path = os.path.expanduser(path)
    vocabs = read_vocabs()

    for name, words in vocabs.items():
        mypath = path + name + '.txt'
        write_vocab(mypath, words)
