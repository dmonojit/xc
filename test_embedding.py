import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utils.util import get_x_y_v_e
from deepwalk import DeepWalk
from node2vec import Node2Vec

def embed(args):
	_, _, _, _, _, V, E = get_x_y_v_e(args.input)

	wv_model = getattr(get_rule(args), 'transform')(E, 'edgedict')

	if args.output:
		wv_model.wv.save_word2vec_format(args.output)

def get_rule(args):
	if args.rule == 'deepwalk':
		return DeepWalk()
	elif args.rule == 'node2vec':
		return Node2Vec()
	else:
		raise NotImplemented

def main():
	parser = ArgumentParser('test_embedding',
                        	formatter_class=ArgumentDefaultsHelpFormatter,
                          	conflict_handler='resolve')

	parser.add_argument('--rule', default='deepwalk', help='deepwalk | node2vec', required=True)

	parser.add_argument('--input', default=None, help='Input train file', required=True)

	parser.add_argument('--output', default=None, help='output file for embeddings')

	# not being used currently
	parser.add_argument('--label_file', default=None, help='Label in words')

	args = parser.parse_args()
	embed(args)


if __name__ == '__main__':
	sys.exit(main())