import sys
import networkx as nx
import datetime
import operator
from functools import reduce
from scipy.sparse import *
from scipy import *

def get_matrices_from_file(filepath):
	with open(filepath) as file:
		content = file.read().splitlines()

	total_points, feature_dm, number_of_labels = map(lambda x: int(x), content.pop(0).split(' '))

	feature_indptr = [0]
	feature_indices = []
	feature_data = []
	
	label_vectors = []

	for row in content:
		row = row.split(' ')
		labels = []

		if row[0] == '':
			row.pop(0)
			label_vectors.append([])
		else:
			label_vectors.append([*map(lambda x: int(x), row.pop(0).split(','))])

		length = len(row)
		for datapoint in row:
			index, value = datapoint.split(':')
			index = int(index)
			value = double(value)
			feature_indices.append(index)
			feature_data.append(value)
		feature_indptr.append(feature_indptr[-1] + length)

	feature_matrix = csr_matrix((feature_data, feature_indices, feature_indptr), dtype=double)


	label_graph = build_label_graph(label_vectors)

	return total_points, feature_dm, number_of_labels, feature_matrix, label_vectors, label_graph

def build_label_graph(label_vectors):
	V = set(reduce(operator.concat, label_vectors))
	E = {}
	for vector in label_vectors:
		for u in range(len(vector)):
			for v in range(u+1, len(vector)):
				t = tuple(sorted([vector[u], vector[v]]))
				if t not in E:
					E[t] = 1
				else:
					E[t] += 1

	G = nx.Graph()
	G.add_edges_from(E.keys())

	return G

if __name__ == '__main__':
	total_points, feature_dm, number_of_labels, feature_matrix, label_vectors, label_graph = get_matrices_from_file(sys.argv[1])
	graph_filepath = sys.argv[2] if len(sys.argv) == 3 else 'label_graph_{}.graphml'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
	nx.write_graphml(label_graph, graph_filepath)
	print('Label graph generated at ' + graph_filepath)
	print(feature_matrix.todense())
