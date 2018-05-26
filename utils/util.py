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

# Deprecated
# def get_matrices_from_file(filepath):
# 	with open(filepath) as file:
# 		content = file.read().splitlines()

# 	total_points, feature_dm, number_of_labels = map(lambda x: int(x), content.pop(0).split(' '))

# 	feature_indptr = [0]
# 	feature_indices = []
# 	feature_data = []
	
# 	label_vectors = []

# 	for row in content:
# 		row = row.split(' ')
# 		labels = []

# 		if row[0] == '':
# 			row.pop(0)
# 			label_vectors.append([])
# 		else:
# 			label_vectors.append([*map(lambda x: int(x), row.pop(0).split(','))])

# 		length = len(row)
# 		for datapoint in row:
# 			index, value = datapoint.split(':')
# 			index = int(index)
# 			value = double(value)
# 			feature_indices.append(index)
# 			feature_data.append(value)
# 		feature_indptr.append(feature_indptr[-1] + length)

# 	feature_matrix = csr_matrix((feature_data, feature_indices, feature_indptr), dtype=double)

# 	label_graph = build_label_graph(label_vectors)

# 	return total_points, feature_dm, number_of_labels, feature_matrix, label_vectors, label_graph

def get_matrices_from_file(filepath, label_filepath):
	with open(filepath, 'r') as file:
		total_points, feature_dm, number_of_labels = map(lambda x: int(x), file.readline().split(' '))
	
	X, Y = load_svmlight_file(filepath, n_features=feature_dm, multilabel=True, offset=1)

	label_graph = build_label_graph(Y, label_filepath)

	return total_points, feature_dm, number_of_labels, X, Y, label_graph

def build_label_graph(Y, label_filepath, level=1):
	list_of_edge_lists = list(map(lambda x: list(itertools.combinations(x, 2)), Y))

	V = set(itertools.chain.from_iterable(Y))
	E = set(itertools.chain.from_iterable(list_of_edge_lists))

	g = nx.Graph()
	g.add_edges_from(list(E))

	# nx.relabel_nodes(g, get_label_dict(label_filepath), copy=False)

	list_v = list(V)
	np.random.shuffle(list_v)
	v = list_v[0]

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
			for uv_tuple in edges:
				sub_g.add_edges_from([uv_tuple])
				bfs_q.put(uv_tuple[1])

	label_dict = get_label_dict(label_filepath)
	def mapping(v):
		return label_dict[v]

	nx.relabel_nodes(sub_g, mapping, copy=False)

	return sub_g

def get_label_dict(label_filepath):
	with open(label_filepath) as file:
		content = file.read().splitlines()

	label_dict = {}
	for i, label in enumerate(content):
		label_dict[i] = label

	return label_dict


# Deprecated
# def build_label_graph(label_vectors):
# 	V = set(reduce(operator.concat, label_vectors))
# 	E = {}
# 	for vector in label_vectors:
# 		for u in range(len(vector)):
# 			for v in range(u+1, len(vector)):
# 				t = tuple(sorted([vector[u], vector[v]]))
# 				if t not in E:
# 					E[t] = 1
# 				else:
# 					E[t] += 1

# 	G = nx.Graph()
# 	G.add_edges_from(E.keys())
# 	for 

# 	return G

if __name__ == '__main__':
	# sample call: python utils/util.py /Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt /Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt
	# '/Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt', '/Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt'
	assert len(sys.argv) >= 3, 'Data file and Label file are required'
	total_points, feature_dm, number_of_labels, feature_matrix, label_vectors, label_graph = get_matrices_from_file(sys.argv[1], sys.argv[2])
	graph_filepath = sys.argv[3] if len(sys.argv) == 4 else 'label_graph_{}.graphml'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
	nx.write_graphml(label_graph, graph_filepath)
	print('Label graph generated at ' + graph_filepath)
	print(feature_matrix.todense())
