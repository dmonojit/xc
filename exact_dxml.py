from utils.util import get_x_y_v_e
from deepwalk import DeepWalk
from sklearn.preprocessing import MultiLabelBinarizer

class ExactDXML(object):
	def __init__(self):
		super()

	def fit(self, train_file, dw_output_file=None):
		_, _, _, _, _, V, E = get_x_y_v_e(train_file)

		wv_model = DeepWalk().transform(E.keys(), 'edgelist')

		# need to work with wv_model.wv.vectors

		if dw_output_file:
			wv_model.wv.save_word2vec_format(dw_output_file)
