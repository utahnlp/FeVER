# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-11-28 13:29:35
# Last modified: 2019-03-22 13:02:21

"""
Transform the text file into XML format with features.
"""

from pathlib import Path
import os
import logging

from tqdm import tqdm

from gel.tool import word2XML
from gel.tool import utils
from feat import Extractor

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)
xfeats = ['word', 'fix', 'n-gram', 'gazetteers',
          'know_fix', 'word_shape', 'special']
# xfeats = ['word']

yfeats = ['word', 'fix', 'n-gram', 'gazetteers',
          'know_fix', 'word_shape', 'special']
# yfeats = ['word']


# xfeats = ['word', 'fix']
# yfeats = ['word', 'fix']

xfeats_extractor = Extractor(funcs=xfeats)
yfeats_extractor = Extractor(funcs=yfeats)


def make_feat_mappings(data, feat_extractor, inv_vocab):
    """Generate the mapping dictionary for features.

    Args:
        data: list(tuple)
        feat_extractor: feature extractor
        inv_vocab: inverse vocabulary. int->str
    """
    feat_mapping = dict()
    for item in tqdm(data):
        words = [inv_vocab[i] for i in item]
        feat = feat_extractor.feats(words)
        for w in feat:
            if w not in feat_mapping:
                feat_mapping[w] = len(feat_mapping)
    return feat_mapping


def extract_feat(data, feat_extractor, feat_mapping, inv_vocab):
    for item in data:
        words = [inv_vocab[i] for i in item]
        feat = feat_extractor.feats(words)
        feat = [w for w in feat if w in feat_mapping]
        # feat = [feat_mapping[w] for w in feat]
        yield feat


def write_examples(labels, contexts, feat_extractor,
                   feat_mapping, inv_vocab, path):
    assert len(labels) == len(contexts)
    label_num = len(inv_vocab)
    feat_num = len(feat_mapping)
    example_num = len(labels)

    word_path = path.parents[0] / (path.name + '.word')
    wordf = open(word_path, 'w', encoding='utf8')
    with open(path, 'w', encoding='utf8') as f:
        s = [example_num, feat_num, label_num]
        s = [str(t) for t in s]
        s = ' '.join(s)
        f.write(s+'\n')
        feat_iter = extract_feat(contexts, feat_extractor,
                                 feat_mapping, inv_vocab)
        for label, feat in tqdm(zip(labels, feat_iter), total=len(labels)):
            assert len(label) != 0
            slabel = [inv_vocab[t] for t in label]
            slabel = ','.join(slabel)
            sfeat = ' '.join(feat)
            s = [slabel, sfeat]
            s = ' '.join(s)
            wordf.write(s+'\n')

            slabel = [str(t) for t in label]
            slabel = ','.join(slabel)
            sfeat = [str(feat_mapping[t]) for t in feat]
            sfeat = [t+':1.0' for t in sfeat]
            sfeat = ' '.join(sfeat)
            s = [slabel, sfeat]
            s = ' '.join(s)
            f.write(s+'\n')


def main(corpus_path, dest_path):
    path = Path(dest_path)
    if not path.exists():
        os.mkdir(path)

    data, vocab = word2XML.generate_training_examples(corpus_path)
    inv_vocab = {v: k for k, v in vocab.items()}
    contexts, labels, freqs = data[0], data[1], data[2]

    words_list = [(t, ) for t in vocab.values()]
    logger.info('Generating context feature mappings...')
    xfeat_mapping = make_feat_mappings(words_list, xfeats_extractor, inv_vocab)
    logger.info('Generating label feature mappings...')
    yfeat_mapping = make_feat_mappings(words_list, yfeats_extractor, inv_vocab)

    logger.info('Generating context features ...')
    context_path = path / 'context_feature_training.txt'
    write_examples(labels, contexts, xfeats_extractor,
                   xfeat_mapping, inv_vocab, context_path)

    logger.info('Generating label features ...')
    label_path = path / 'label_feature_training.txt'
    write_examples(words_list, words_list, yfeats_extractor,
                   yfeat_mapping, inv_vocab, label_path)

    logger.info('Writing down the frequency file ...')
    freq_path = path / 'frequency.txt'
    utils.write_freq(freq_path, freqs)

    logger.info('Writing down the vocabulary...')
    vocab_path = path / 'vocabulary.txt'
    utils.write_vocab(vocab_path, vocab)

    logger.info('Writing down the context feature mappings...')
    xfeat_path = path / 'context_feature_mapping.txt'
    utils.write_vocab(xfeat_path, xfeat_mapping)

    logger.info('Writing down the label feature mappings...')
    yfeat_path = path / 'label_feature_mapping.txt'
    utils.write_vocab(yfeat_path, yfeat_mapping)


def label_features(feat_extractor, vocab_path,
                   feat_mapping_path, example_path):
    vocab = utils.read_vocab(vocab_path)
    inv_vocab = {v: k for k, v in vocab.items()}

    feat_mapping = utils.read_vocab(feat_mapping_path)
    words_list = [(t, ) for t in vocab.values()]
    logger.info('Generating features ...')
    write_examples(words_list, words_list, feat_extractor,
                   feat_mapping, inv_vocab, example_path)


if __name__ == '__main__':
    # corpus_path = '../../embedding_data/corpus/ten_percent.txt'
    # corpus_path = '../../embedding_data/corpus/toy2/toy2.txt'
    # main(corpus_path, '../../embedding_data/toy2/')

    # vocab_path = '../../embedding_data/three_word/large_vocabulary.txt'
    # feat_mapping_path = ('../../embedding_data/three_word/'
    #                      'label_feature_mapping.txt')
    # example_path = ('../../embedding_data/three_word/'
    #                 'large_vocab_label_features.txt')

    vocab_path = '../../embedding_data/RST/RST_vocab.txt'
    feat_mapping_path = ('../../embedding_data/ten_all/'
                         'context_feature_mapping.txt')
    example_path = ('../../embedding_data/RST/'
                    'context_features.txt')

    example_path = Path(example_path)
    label_features(xfeats_extractor, vocab_path, feat_mapping_path,
                   example_path)
