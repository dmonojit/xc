from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from gensim.models import KeyedVectors

from deepwalk import DeepWalk
from node2vec import Node2Vec
from utils.util import get_x_y_v_e, get_label_dict

class Compare(object):

    def __init__(self, label_file, emb_model):
        self.wv_train_file_path = '/Users/monojitdey/Downloads/GoogleNews-vectors-negative300.bin'

        self.emb_model = emb_model
        self.wv_model = self.get_wv_model()
        self.label_dict = get_label_dict(label_file)
        self.number_of_labels = len(self.label_dict.keys())

        self.similarity_matrix_emb = [[None for i in range(self.number_of_labels)] for i in range(self.number_of_labels)]
        self.similarity_matrix_wv = [[None for i in range(self.number_of_labels)] for i in range(self.number_of_labels)]
        self.comparison_tuples = []

    def get_wv_model(self):
        print('Learning word2vec...')
        return KeyedVectors.load_word2vec_format(self.wv_train_file_path, binary=True)

    def generate_similarities(self):
        print('Generating similarities...')
        for i in range(self.number_of_labels):
            for j in range(i+1, self.number_of_labels):
                self.similarity_matrix_wv[i][j] = self.similarity_matrix_wv[j][i] = self.wv_model.similarity(self.label_dict[i], self.label_dict[j])
                self.similarity_matrix_emb[i][j] = self.similarity_matrix_emb[j][i] = self.emb_model.similarity(str(i), str(j))

    def generate_comparisons(self):
        print('Generating comparisons...')
        for i in range(self.number_of_labels):
            for j in range(i+1, self.number_of_labels):
                score_diff = abs(self.similarity_matrix_wv[i][j] - self.similarity_matrix_emb[i][j])
                self.comparison_tuples.append((i, j, score_diff))

    def rank_comparisons(self):
        print('Ranking...')
        self.comparison_tuples = sorted(self.comparison_tuples, key = lambda x: x[2])

    def invoke(self):
        self.generate_similarities()
        self.generate_comparisons()
        self.rank_comparisons()

def main(args):
    _,_,_,X_tr,Y_tr,V_tr,E_tr = get_x_y_v_e(args.filepath)

    print('Learning embeddings with {}...'.format(args.embedding))

    if args.embedding == 'deepwalk':
        ## DeepWalk default values: number_walks=10,representation_size=64,seed=0,walk_length=40,window_size=5,workers=1
        label_emb = DeepWalk().transform(E_tr,'edgedict')
    elif args.embedding == 'node2vec':
        ## Node2Vec default values: num_walks=10,dimensions=64,walk_length=40,window_size=5,workers=1,p=1,q=1,
        ## weighted=False,directed=False,iter=1
        label_emb = Node2Vec().transform(E_tr,'edgedict')
    else:
        raise NotImplemented

    label_emb_wv = label_emb.wv

    print('Calling compare...')

    compare = Compare(args.labelfile, label_emb_wv)
    compare.invoke()

    import pdb; pdb.set_trace()

    print('end')


if __name__ == '__main__':
    parser = ArgumentParser("Embedding comparator",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            epilog="""
                                    Example: python compare.py
                                    --filepath /Users/monojitdey/ml/datasets/Amazon/AmazonCat/amazonCat_test.txt 
                                    --labelfile /Users/monojitdey/ml/datasets/Amazon/AmazonCat-13K_mappings/AmazonCat-13K_label_map.txt \n
                                   """)
    
    parser.add_argument('--filepath', help='Path to train file',required=True)
    parser.add_argument('--labelfile', help='Path to label file',required=True)
    parser.add_argument('--embedding', help='Emdedding rule - deepwalk|node2vec', default='deepwalk')

    args = parser.parse_args()
    main(args)