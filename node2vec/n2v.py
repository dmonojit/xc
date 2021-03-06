'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx

from . import graph
from gensim.models import Word2Vec


class Node2Vec(object):
	def __init__(self, **kwargs):
		self.dimensions = kwargs.get('dimensions', 64)
		self.walk_length = kwargs.get('walk_length', 40)
		self.num_walks = kwargs.get('num_walks', 10)
		self.window_size = kwargs.get('window_size', 5)
		self.iter = kwargs.get('iter', 1)
		self.workers = kwargs.get('workers', 1)
		self.p = kwargs.get('p', 1)
		self.q = kwargs.get('q', 1)
		self.weighted = kwargs.get('weighted', False)
		self.directed = kwargs.get('directed', False)
		self.output = kwargs.get('output', None)

	def transform(self, input_data, input_format):
		'''
		Pipeline for representational learning for all nodes in a graph.
		'''
		print('Preparing graph...')
		nx_G = self.read_graph(input_data, input_format)
		G = graph.Graph(nx_G, self.directed, self.p, self.q)

		print('Preprocessing transition probabilities...')
		G.preprocess_transition_probs()
		
		print('Walking...')
		walks = G.simulate_walks(self.num_walks, self.walk_length)
		
		print('Training...')
		return self.learn_embeddings(walks)

	def read_graph(self, input_data, input_format):
		'''
		Reads the input network in networkx.
		'''
		if input_format == 'edgedict':
			self.weighted = True
			edgelist = [edge + ({'weight': weight},) for edge, weight in input_data.items()]
			G = nx.Graph()
			G.add_edges_from(edgelist)
		else:
			if self.weighted:
				G = nx.read_edgelist(input_data, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
			else:
				G = nx.read_edgelist(input_data, nodetype=int, create_using=nx.DiGraph())
				for edge in G.edges():
					G[edge[0]][edge[1]]['weight'] = 1

		if not self.directed:
			G = G.to_undirected()

		return G

	def learn_embeddings(self, walks):
		'''
		Learn embeddings by optimizing the Skipgram objective using SGD.
		'''
		walks = [[*map(str, walk)] for walk in walks]
		model = Word2Vec(walks, size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=self.workers, iter=self.iter)
		
		if self.output:
			model.wv.save_word2vec_format(self.output)
		
		return model
