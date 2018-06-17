import sys
import networkx as nx
import datetime
import operator
from functools import reduce
from scipy.sparse import *
from scipy import *
from sklearn.datasets import load_svmlight_file
import itertools
import numpy as np
from queue import Queue
from  collections import defaultdict
import matplotlib.pyplot as plt

def get_x_and_y(filepath):
	with open(filepath, 'r') as file:
		total_points, feature_dm, number_of_labels = map(lambda x: int(x), file.readline().split(' '))
	
	X, Y = load_svmlight_file(filepath, n_features=feature_dm, multilabel=True, offset=1)

	return total_points, feature_dm, number_of_labels, X, Y


def get_matrices_from_file(filepath, label_filepath):
	with open(filepath, 'r') as file:
		total_points, feature_dm, number_of_labels = map(lambda x: int(x), file.readline().split(' '))
	
	total_points, feature_dm, number_of_labels, X, Y = get_x_and_y(filepath)

	label_graph = build_label_graph(Y, label_filepath)

	return total_points, feature_dm, number_of_labels, X, Y, label_graph

def get_x_y_v_e(filepath):
	total_points, feature_dm, number_of_labels, X, Y = get_x_and_y(filepath)

	list_of_edge_lists = list(map(lambda x: list(itertools.combinations(x, 2)), Y))

	V = list(set(itertools.chain.from_iterable(Y)))
	# E = set(itertools.chain.from_iterable(list_of_edge_lists))
	E = defaultdict(lambda: 0)
	for e in itertools.chain.from_iterable(list_of_edge_lists):
		sorted_edge = tuple(sorted(e))
		E[e] += 1

	return total_points, feature_dm, number_of_labels, X, Y, V, E

def get_subgraph(filepath, label_filepath, level=1, max_edges_per_node=None, root_node=None):
	# total_points: total number of data points
	# feature_dm: number of features per datapoint
	# number_of_labels: total number of labels
	# X: feature matrix of dimension total_points * feature_dm
	# Y: list of size total_points. Each element of the list containing labels corresponding to one datapoint
	# V: list of all labels (nodes)
	# E: dict of edge tuple -> weight, eg. {(1, 4): 1, (2, 7): 3}
	total_points, feature_dm, number_of_labels, X, Y, V, E = get_x_y_v_e(filepath)

	# get a dict of label -> textual_label
	label_dict = get_label_dict(label_filepath)

	# an utility function to relabel nodes of upcoming graph with textual label names
	def mapping(v):
		v = int(v)
		if v in label_dict:
			return label_dict[v]

		return str(v)

	# build a unweighted graph of all edges
	g = nx.Graph()
	g.add_edges_from(E.keys())

	# Below section will try to build a smaller subgraph from the actual graph for visualization
	
	if root_node is None:
	# select a random vertex to be the root
		np.random.shuffle(V)
		v = V[0]
	else:
		v = root_node

	# two files to write the graph and label information
	label_info_filepath = 'samples/label_info_{}.txt'.format(str(int(v)) + '_' + mapping(v)).replace(' ', '')
	label_graph_filepath = 'samples/label_graph_{}.graphml'.format(str(int(v)) + '_' + mapping(v)).replace(' ', '')
	label_info_file = open(label_info_filepath, 'w')

	# build the subgraph using bfs
	bfs_q = Queue()
	bfs_q.put(v)
	bfs_q.put(0)
	node_check = {}

	sub_g = nx.Graph()
	l = 0
	while not bfs_q.empty() and l <= level:
		v = bfs_q.get()
		if v == 0:
			l += 1
			bfs_q.put(0)
			continue
		elif node_check.get(v, True):
			node_check[v] = False
			edges = list(g.edges(v))
			label_info_file.write('\nNumber of edges: ' + str(len(edges)) + ' for node: ' + mapping(v) + '[' + str(v) + ']' + '\n')
			if max_edges_per_node is not None and len(edges) > max_edges_per_node:
				label_info_file.write('Ignoring node in graph: ' + mapping(v) + '\n')
				continue
			for uv_tuple in edges:
				edge = tuple(sorted(uv_tuple))
				sub_g.add_edge(edge[0], edge[1], weight=E[edge])
				bfs_q.put(uv_tuple[1])
		else:
			continue

	# relabel the nodes to reflect textual label
	nx.relabel_nodes(sub_g, mapping, copy=False)

	label_info_file.close()
	nx.write_graphml(sub_g, label_graph_filepath)

	print('Label info generated at ' + label_info_filepath)
	print('Label graph generated at ' + label_graph_filepath)

	return sub_g

def get_label_dict(label_filepath):
	if label_filepath is None:
		return {}

	try:
		with open(label_filepath, 'r') as file:
			content = file.read().splitlines()
	except:
		with open(label_filepath, 'r', encoding='latin-1') as file:
			content = file.read().splitlines()

	label_dict = {}
	for i, label in enumerate(content):
		label_dict[i] = str(label)

	return label_dict

if __name__ == '__main__':
	# sample call: python utils/util.py /Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt /Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt
	assert len(sys.argv) >= 2, 'Data file is required'
	if len(sys.argv) < 3:
		print('Label file is not provided, graph will show numeric labels only')
	if len(sys.argv) == 4:
		root_node = int(sys.argv[3])
	else:
		root_node = None
	label_graph = get_subgraph(sys.argv[1], sys.argv[2], level=1, max_edges_per_node=200, root_node=root_node)
	# nx.draw(label_graph)
	# plt.show()
