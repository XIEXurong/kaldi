#!/usr/bin/env python

import os
import sys
import numpy as np

dir = sys.argv[1]
new_word_list = sys.argv[2]

word2newidx = {}
oldidx2word = []
oldidx2newidx = []

old_word_list = os.path.join(dir, 'config/words.txt')
old_word_feats = os.path.join(dir, 'word_feats.txt')
new_word_feats = os.path.join(dir, 'word_feats_new.txt')

assert (os.path.isfile(old_word_list) and os.path.isfile(old_word_feats) and os.path.isfile(new_word_list))

f = open(old_word_list, 'r', encoding='utf-8')

for line in f:
    word2idx = line.split()
    assert (len(word2idx) == 2)
    word = word2idx[0]
    idx = int(word2idx[1])
    assert (len(oldidx2word) == idx)
    oldidx2word.append(word)

print("Read old list with",len(oldidx2word),"words.")

f.close()

f = open(new_word_list, 'r', encoding='utf-8')

for line in f:
    word2idx = line.split()
    assert (len(word2idx) == 2)
    word = word2idx[0]
    idx = word2idx[1]
    word2newidx[word] = idx

print("Read old list with",len(word2newidx),"words.")

f.close()

for idx in range(len(oldidx2word)):
    try:
        oldidx2newidx.append(word2newidx[oldidx2word[idx]])
    except:
        oldidx2newidx.append("-1")

f = open(old_word_feats, 'r', encoding='utf-8')

new_feats = [str(i) for i in (np.zeros(len(word2newidx))-1)]

for line in f:
    feats = line.split()
    idx = oldidx2newidx[int(feats[0])]
    if idx is not "-1":
        feats[0] = idx
        new_feats[int(idx)] = " ".join(feats)+"\n"

f.close()

f = open(new_word_feats, 'w', encoding='utf-8')

for idx in range(len(new_feats)):
    f.write(new_feats[idx])

f.close()
