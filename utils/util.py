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
import matplotlib.pyplot as plt

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
	label_dict = get_label_dict(label_filepath)

	def mapping(v):
		v = int(v)
		if v in label_dict:
			return label_dict[v]

		return str(v)

	list_of_edge_lists = list(map(lambda x: list(itertools.combinations(x, 2)), Y))

	V = set(itertools.chain.from_iterable(Y))
	E = set(itertools.chain.from_iterable(list_of_edge_lists))

	g = nx.Graph()
	g.add_edges_from(list(E))

	# nx.relabel_nodes(g, get_label_dict(label_filepath), copy=False)

	list_v = list(V)
	np.random.shuffle(list_v)
	v = list_v[0]
	label_info_filepath = 'samples/label_info_{}.txt'.format(str(int(v)) + '_' + mapping(v)).replace(' ', '')
	label_graph_filepath = 'samples/label_graph_{}.graphml'.format(str(int(v)) + '_' + mapping(v)).replace(' ', '')

	label_info_file = open(label_info_filepath, 'w')

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
			label_info_file.write('\nNumber of edges: ' + str(len(edges)) + ' for node: ' + mapping(v) + '\n')
			# if len(edges) > 100:
			# 	label_info_file.write('Ignoring node in graph: ' + mapping(v) + '\n')
			# 	continue
			for uv_tuple in edges:
				sub_g.add_edges_from([uv_tuple])
				bfs_q.put(uv_tuple[1])

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
	assert len(sys.argv) >= 2, 'Data file is required'
	if len(sys.argv) < 3:
		print('Label file is not provided, graph will show numeric labels only')
	total_points, feature_dm, number_of_labels, feature_matrix, label_vectors, label_graph = get_matrices_from_file(sys.argv[1], sys.argv[2])
	nx.draw(label_graph)
	plt.show()
	# print(feature_matrix.todense())
