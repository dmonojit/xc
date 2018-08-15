#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Tools for Word2Vec operations
__description__ :
__project__     : my_modules
__author__      : 'Samujjwal Ghosh'
__version__     :
__date__        : June 2018
__copyright__   : "Copyright (c) 2018"
__license__     : "Python"; (Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html)

__classes__     :

__variables__   :

__methods__     :

TODO            : 1.
"""

import os
import numpy as np
from collections import OrderedDict

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import my_modules as mm


def open_word2vec(w2v_bin_path,binary=True):
    # model = None
    try:
        model = KeyedVectors.load_word2vec_format(w2v_bin_path,binary=True)
    except Exception as e: ## Loading a different format.
        print('Loading original word2vec format failed. Trying Gensim format.')
        model = KeyedVectors.load(w2v_bin_path,binary=True)
    return model


def init_w2v(w2v_bin_path='/home/Embeddings/',w2v_bin_file='GoogleNews-vectors-negative300.bin'): ## Alternate: w2v_bin_file=glove.840B.300d.txt
    try:
        w2v = open_word2vec(os.path.join(w2v_bin_path,w2v_bin_file))
    except Exception as e:
        print("Failed to load Word2Vec binary file from:",os.path.join(w2v_bin_path,w2v_bin_file))
        print('Failure reason:',e)
    return w2v


def gen_txt_w2v(train,w2v):
    train_vec = OrderedDict()
    for id,val in train.items():
        s_vec = np.zeros(300)
        for word in val['parsed_tweet'].split(" "):
            if word in w2v.wv.vocab:
                # train_vec[id][word] = w2v[word].tolist()
                s_vec = np.add(s_vec, w2v[word])
            else:
                pass
                # print("Word [",word,"] not in vocabulary")
            # print("\n")
        train_vec[id]=s_vec
    return train_vec


def cosine_sim(w2v,w1,w2):
    return w2v.similarity(w1,w2)


def find_sim(w2v,word,k=5):
    #print("Finding similar words of:",word)
    w2v_words = []
    if word in w2v.wv.vocab:
        w2v_words = w2v.most_similar(positive=[word],negative=[],topn=k)
        #for term,val in list(w2v_words):
         #   word_list = word_list + [term]
    return w2v_words


def find_sim_list(w2v,words,k=5):

    for word in words:
        words = words + find_sim(w2v,word,k)

    words = mm.remove_dup_list(words, case=True)
    return words[0:k]


def expand_tweet(w2v,tweet,c=3):
    new_tweet = []
    for word in tweet.split(" "):
        new_tweet= new_tweet+[word]
        w2v_words = find_sim(w2v,word,c)
        #if word in w2v.vocab:
         #   w2v_words=w2v.most_similar(positive=[word], negative=[], topn=c)
        for term,val in w2v_words:
            new_tweet= new_tweet+[term]
    return new_tweet


def expand_tweets(w2v,dict):
    # print("Method: expand_tweets(dict)")
    for id,val in dict.items():
        val['expanded_tweet'] = "".join(expand_tweet(w2v,val['parsed_tweet']))
    return dict


def create_w2v(corpus,size=1000,window=5,min_count=3,workers=10):
    w2v = Word2Vec(corpus,size,window,min_count,workers)
    # print(w2v)
    # print(type(w2v))
    return w2v


def main():
    pass


if __name__ == "__main__": main()