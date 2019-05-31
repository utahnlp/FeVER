# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-11-29 12:45:57
# Last modified: 2018-12-04 13:42:38

"""
Generate the vocabulary based on different datasets.
"""
import xml.etree.ElementTree as ET
import os


def parseXML(path):
    """Parse one yahoo document.
    """
    reval = set()
    s = 'Reading yahoo: {a}'.format(a=str(path))
    print(s)
    with open(path, encoding='utf8') as f:
        text = f.readlines()
        # Broken data
        if text[-2].strip() != '</TEXT>':
            text.insert(-1, '</TEXT>')
        text = ''.join(text)
        text = '<ROOT>' + text + '</ROOT>'
        tree = ET.fromstring(text)
        doclist = tree.findall('DOC')
        for doc in doclist:
            docno = doc.find('DOCNO')
            docno = docno.text.strip()
            text = doc.find('TEXT')
            text = text.text.strip()
            reval |= set(text.split())
    return reval


def YahooSet():
    path = ('../../text-classification/yahoo.answers/'
            'yahoo.data.raw/')
    names = os.listdir(path)
    text_set = set()
    print('Reading yahoo dataset...')
    for name in names:
        file_path = os.path.join(path, name)
        data = parseXML(file_path)
        text_set |= data
    return text_set


############################################################
# 20News
############################################################
def NewsSet():
    path = ('../../text-classification/20news/20news-processed/'
            '20ng-test-all-terms.txt')
    reval = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            labels = s[0].split('.')
            reval |= set(labels)
            reval |= set(s[1:])

    path = ('../../text-classification/20news/20news-processed/'
            '20ng-train-all-terms.txt')
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            labels = s[0].split('.')
            reval |= set(labels)
            reval |= set(s[1:])
    return reval


############################################################
# Batting
############################################################
def read_battig(path):
    s = 'Reading battig: {a}'.format(a=str(path))
    print(s)
    reval = set()
    with open(path, encoding='utf8') as f:
        next(f)
        for line in f:
            s = line.strip().split(',')
            word = s[0]
            word = ''.join(word.split())
            reval.add(word)
    return reval


def BattigSet():
    path = '~/scr/web_data/categorization/EN-BATTIG'
    path = os.path.expanduser(path)
    names = os.listdir(path)
    reval = set()
    print('Reading battig set...')
    for name in names:
        file_path = os.path.join(path, name)
        reval |= read_battig(file_path)
    return reval


############################################################
# Reading MEN
############################################################
def MENSet():
    path = '~/scr/web_data/similarity/EN-MEN-LEM.txt'
    path = os.path.expanduser(path)
    reval = set()
    s = 'Reading MEN set: {a}'.format(a=str(path))
    print(s)
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            w1 = s[0].split('-')[0]
            w2 = s[1].split('-')[0]
            reval.add(w1)
            reval.add(w2)
    return reval


############################################################
# Reading RW
############################################################
def RWSet():
    path = '~/scr/web_data/similarity/EN-RW.txt'
    path = os.path.expanduser(path)
    reval = set()
    s = 'Reading RW set: {a}'.format(a=str(path))
    print(s)
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            reval.add(s[0])
            reval.add(s[1])
    return reval


############################################################
# Reading SimLex999
############################################################
def SimLexSet():
    path = '~/scr/web_data/similarity/EN-SIM999.txt'
    path = os.path.expanduser(path)
    reval = set()
    s = 'Reading SimLex999 set: {a}'.format(a=str(path))
    print(s)
    with open(path, encoding='utf8') as f:
        next(f)
        for line in f:
            s = line.strip().split()
            reval.add(s[0])
            reval.add(s[1])
    return reval


############################################################
# Reading Google
############################################################
def read_google(path):
    reval = set()
    s = 'Reading google set: {a}'.format(a=str(path))
    print(s)
    with open(path, encoding='utf8') as f:
        for line in f:
            if line.startswith(':'):
                continue
            else:
                s = line.strip().split()
                for w in s:
                    reval.add(w)
    return reval


def GoogleSet(typo=False):
    path = '~/scr/web_data/analogy/EN-GOOGLE'
    path = os.path.expanduser(path)
    names = os.listdir(path)
    reval = set()
    print('Reading google set...')
    if typo is False:
        names = [name for name in names if 'degree' not in name]
    print(names)
    for name in names:
        file_path = os.path.join(path, name)
        reval |= read_google(file_path)
    return reval


############################################################
# Reading MSR
############################################################
def read_MSR(path):
    s = 'Reading MSR: {a}'.format(a=str(path))
    print(s)
    reval = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            reval.add(s[0])
            reval.add(s[1])
            reval.add(s[2])
            reval.add(s[-1])
    return reval


def MSRSet(typo=False):
    path = '~/scr/web_data/analogy/EN-MSR'
    path = os.path.expanduser(path)
    names = os.listdir(path)
    reval = set()
    print('Reading MSR set...')
    if typo is False:
        names = [name for name in names if 'degree' not in name]
    print(names)
    for name in names:
        file_path = os.path.join(path, name)
        reval |= read_MSR(file_path)
    return reval


def read_vocab(path):
    reval = dict()
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split('$:$')
            word = s[0]
            ID = int(s[1])
            reval[word] = ID
    return reval


def read_embeds(path):
    reval = set()
    with open(path, encoding='utf8') as f:
        next(f)
        for line in f:
            s = line.strip().split()
            word = s[0]
            reval.add(word)
    return reval


def ReadDict(train_vocab_path):
    words = read_embeds(train_vocab_path)
    words |= YahooSet()
    words |= NewsSet()
    words |= BattigSet()
    words |= MENSet()
    words |= RWSet()
    words |= SimLexSet()
    words |= GoogleSet(typo=True)
    words |= MSRSet(typo=True)

    mapping = dict()
    for i, word in enumerate(words):
        mapping[word] = str(i)
    return mapping


def write_vocab(path, vocab):
    with open(path, 'w', encoding='utf8') as f:
        for word, ID in vocab.items():
            s = [str(word), str(ID)]
            s = '$:$'.join(s)
            f.write(s+'\n')


if __name__ == '__main__':
    # train_vocab_path = '../../embedding_data/ten_word/vocabulary.txt'
    train_vocab_path = '~/scr/word2vec/embeddings/all_corpus_cbow.txt'
    train_vocab_path = os.path.expanduser(train_vocab_path)
    vocab = ReadDict(train_vocab_path)
    out_path = '../../embedding_data/three_word/large_vocabulary.txt'
    write_vocab(out_path, vocab)
