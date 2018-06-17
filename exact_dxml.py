from utils.util import get_label_edges
from deepwalk import DeepWalk
from sklearn.preprocessing import MultiLabelBinarizer

class ExactDXML(object):
	def __init__(self):
		super()

	def fit(self, train_file):
		X, Y, label_nodes, label_edges = get_label_edges(train_file)

		import pdb; pdb.set_trace()

		mlb = MultiLabelBinarizer()
		Y_multi_hot_encoded = mlb.fit_transform(Y)

		wv_model = DeepWalk().transform(label_edges, 'edgelist')

		# need to work with wv_model.wv.vectors

		