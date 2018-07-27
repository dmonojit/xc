#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

from . import graph
from . import walks as serialized_walks
from gensim.models import Word2Vec
from .skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class DeepWalk(object):
    def __init__(self, **kwargs):
        self.number_walks = kwargs.get('number_walks', 10)
        self.representation_size = kwargs.get('representation_size', 64)
        self.seed = kwargs.get('seed', 0)
        self.walk_length = kwargs.get('walk_length', 40)
        self.window_size = kwargs.get('window_size', 5)
        self.workers = kwargs.get('workers', 1)
        self.max_memory_data_size = kwargs.get('max_memory_data_size', 1000000000)
        self.output = kwargs.get('output', None)

        self.vertex_freq_degree = False

    def transform(self, input_data, input_format, undirected=True):
        if input_format == "adjlist":
            raise NotImplementedError('adjlist is not suppoted yet')
            # G = graph.load_adjacencylist(input_data, undirected=undirected)
        elif input_format == "edgelist":
            G = graph.load_edgelist(input_data, undirected=undirected)
        else:
            raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist'" % input_format)

        print("Number of nodes: {}".format(len(G.nodes())))

        num_walks = len(G.nodes()) * self.number_walks

        print("Number of walks: {}".format(num_walks))

        data_size = num_walks * self.walk_length

        print("Data size (walks*length): {}".format(data_size))

        if data_size < self.max_memory_data_size:
            print("Walking...")
            walks = graph.build_deepwalk_corpus(G, num_paths=self.number_walks,
                path_length=self.walk_length, alpha=0, rand=random.Random(self.seed))
            print("Training...")
            model = Word2Vec(walks, size=self.representation_size, window=self.window_size, min_count=0, sg=1, hs=1, workers=self.workers)
        else:
            print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, self.max_memory_data_size))
            print("Walking...")

            walks_filebase = self.output + ".walks"
            walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=self.number_walks,
                path_length=self.walk_length, alpha=0, rand=random.Random(self.seed),
                num_workers=self.workers)

            print("Counting vertex frequency...")
            if not self.vertex_freq_degree:
                vertex_counts = serialized_walks.count_textfiles(walk_files, self.workers)
            else:
                # use degree distribution for frequency in tree
                vertex_counts = G.degree(nodes=G.iterkeys())

            print("Training...")
            walks_corpus = serialized_walks.WalksCorpus(walk_files)
            model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                             size=self.representation_size,
                             window=self.window_size, min_count=0, trim_rule=None, workers=self.workers)

        if self.output:
            model.wv.save_word2vec_format(self.output)
            
        return model
