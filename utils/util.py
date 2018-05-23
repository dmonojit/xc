import sys
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

	return total_points, feature_dm, number_of_labels, feature_matrix, label_vectors

if __name__ == '__main__':
	total_points, feature_dm, number_of_labels, feature_matrix, label_vectors = get_matrices_from_file(sys.argv[1])
	print(feature_matrix.todense())	
