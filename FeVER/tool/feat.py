# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-08-17 12:52:41
# Last modified: 2018-12-07 13:30:51

"""
Extract prefix and suffix features.
"""
import os

from nltk import ngrams
import enchant


FIX_SIZE = 3

MIN_SUB = 3
MAX_SUB = 6


def load_gaze(path):
    names = os.listdir(path)
    names = [t for t in names if t.endswith('.gz') is not True]
    reval = dict()
    for name in names:
        mypath = os.path.join(path, name)
        container = set()
        with open(mypath, encoding='utf8') as f:
            for line in f:
                s = line.strip()
                container.add(s)
        reval[name] = container
    return reval


def load_quirk(path):
    prefix = dict()
    suffix = dict()
    with open(path, encoding='utf8') as f:
        for line in f:
            if len(line.strip()) != 0:
                s = line.strip().split('|')
                s = [t.strip() for t in s]
                fix, label = s[0], s[1]
                if fix.endswith('-') is True:
                    prefix[fix[:-1].lower()] = label
                elif fix.startswith('-') is True:
                    suffix[fix[1:].lower()] = label
                else:
                    raise Exception('Uncognized prefix or suffix !')
    return prefix, suffix


class Extractor:
    def __init__(self, funcs=['word'], index=False):
        self.gaz = load_gaze('gazetteers/')
        self.pre_quirk, self.suf_quirk = load_quirk('Quirk.txt')
        self.index = index
        self.cache = dict()
        self.spell = enchant.Dict('en_US')
        self.funcs_names = funcs

        self.funcs = []
        if 'word' in funcs:
            self.funcs.append(self._word)
        if 'fix' in funcs:
            self.funcs.append(self._pre_suf_fix)
        if 'n-gram' in funcs:
            self.funcs.append(self._substring)
        if 'gazetteers' in funcs:
            self.funcs.append(self._gazetteers)
            self.funcs.append(self._typo)
        if 'know_fix' in funcs:
            self.funcs.append(self._known_fix)
        if 'word_shape' in funcs:
            self.funcs.append(self._word_shape)
        if 'special' in funcs:
            self.funcs.append(self._special)
        self.funcs.append(self._default)

    def feats(self, inputs):
        """
        Args:
            inputs: list(str)
        """
        con = []
        for i, word in enumerate(inputs):
            if word not in self.cache:
                feats = []
                for fun in self.funcs:
                    feats += fun(word, 0)
                self.cache[word] = '$:$'.join(feats)
            else:
                feats = self.cache[word]
                feats = feats.split('$:$')
            con += feats
        return con

    def _word(self, word, index):
        if self.index is True:
            return[''.join(['<', word, '>', '#', str(index)])]
        else:
            return [''.join(['<', word, '>'])]

    def _pre_suf_fix(self, word, index):
        """Extract all prefix and suffix from given word.

        Args:
            word: str
        Returns:
            list(str)
        """
        data = []
        for i in range(1, FIX_SIZE+1):
            w = word[:i].strip()
            w = '#'.join([str(index), str(i), w])
            data.append(w)
        for i in range(-1, -FIX_SIZE-1, -1):
            w = word[i:].strip()
            w = '#'.join([str(index), str(i), w])
            data.append(w)
        return data

    def _substring(self, word, index):
        reval = []
        for size in range(MIN_SUB, MAX_SUB):
            subs = list(ngrams(word, size))
            subs = [''.join(t) for t in subs]
            subs = [''.join(['n-grams@', t]) for t in subs]
            reval += subs
        return reval

    def _gazetteers(self, word, index):
        reval = []
        for name, container in self.gaz.items():
            if word in container:
                s = ['gaz@',  name]
                s = ''.join(s)
                reval.append(s)
        return reval

    def _known_fix(self, word, index):
        reval = []
        lower = word.lower()
        for key in self.pre_quirk.keys():
            if lower.startswith(key):
                s = ['quirk@', self.pre_quirk[key]]
                s = ''.join(s)
                reval.append(s)
        for key in self.suf_quirk.keys():
            if lower.endswith(key):
                s = ['quirk@', self.suf_quirk[key]]
                s = ''.join(s)
                reval.append(s)
        return reval

    def _word_shape(self, word, index):
        con = []
        # Mapping
        for char in word:
            if char.isupper() is True:
                con.append('X')
            elif char.islower() is True:
                con.append('x')
            elif char.isdecimal() is True:
                con.append('d')
            else:
                con.append('-')
        # Slice
        j = 0
        que = []
        for i in range(len(con)):
            if con[i] == con[j]:
                i += 1
            else:
                que.append(con[j:i])
                j = i
        que.append(con[j:i+1])

        # Shrink
        reval = []
        for item in que:
            if len(item) > 3:
                reval.append(item[0]*3)
            else:
                reval.append(''.join(item))
        s = ''.join(reval)
        s = ['shape@', s]
        s = ''.join(s)
        return [s]

    def _special(self, word, index):
        reval = []
        if word.isalnum():
            s = ['spec@', 'alphanumeric']
            s = ''.join(s)
            reval.append(s)
        if word.isalpha():
            s = ['spec@', 'alpha']
            s = ''.join(s)
            reval.append(s)
        if word.isdecimal():
            s = ['spec@', 'decimal']
            s = ''.join(s)
            reval.append(s)
        if word.isdigit():
            s = ['spec@', 'digit']
            s = ''.join(s)
            reval.append(s)
        if word.islower():
            s = ['spec@', 'lower']
            s = ''.join(s)
            reval.append(s)
        if word.isnumeric():
            s = ['spec@', 'numeric']
            s = ''.join(s)
            reval.append(s)
        if word.isprintable():
            s = ['spec@', 'printable']
            s = ''.join(s)
            reval.append(s)
        if word.isupper():
            s = ['spec@', 'cased']
            s = ''.join(s)
            reval.append(s)
        if word.istitle():
            s = ['spec@', 'title']
            s = ''.join(s)
            reval.append(s)
        return reval

    def _typo(self, word, index):
        reval = []
        cond2 = word.isalpha() is True
        cond1 = self.spell.check(word) is False
        if cond1 and cond2:
            closes = self.spell.suggest(word)
            if len(closes) == 0:
                return []
            close = closes[0]
            if close not in self.cache:
                reval += self._word(close, 0)
                reval += self._pre_suf_fix(close, 0)
                reval += self._substring(close, 0)
                reval += self._gazetteers(close, 0)
                reval += self._known_fix(close, 0)
                reval += self._word_shape(close, 0)
                reval += self._special(close, 0)
            else:
                tmp = self.cache[close]
                reval += tmp.split('$:$')
        return reval

    def _default(self, word, index):
        return ['default@null']


if __name__ == '__main__':
    xfeats = ['word', 'fix', 'n-gram', 'gazetteers',
              'know_fix', 'word_shape', 'special']
    extractor = Extractor(xfeats)

    context = 'China'.split()
    feats = extractor.feats(context)
    print(context)
    print(feats)

    context = 'Xhina'.split()
    feats = extractor.feats(context)
    print(context)
    print(feats)

    # spell = SpellChecker()
    # words = ['Google', 'Xhina', 'cChina', 'Berlin', 'Bnrlin']
    # words = [t.lower() for t in words]
    # misspelled = spell.unknown(words)
    # print(misspelled)
