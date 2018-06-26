import sys,os
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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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

def get_subgraph(filepath,label_filepath,level=1,subgraph_count=5,ignore_deg=None,root_node=None):
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

    from networkx.readwrite import json_graph

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

    subgraph_lists = []
    for node in range(subgraph_count):
        if root_node is None:
        # select a random vertex to be the root
            np.random.shuffle(V)
            v = V[0]
        else:
            v = root_node

        # two files to write the graph and label information
        label_info_filepath = 'samples/Info[{}].txt'.format(str(int(v)) + '-' + mapping(v)).replace(' ', '_')
        label_graph_filepath = 'samples/G[{}].graphml'.format(str(int(v)) + '-' + mapping(v)).replace(' ', '_')
        label_graph_el = 'samples/E[{}].el'.format(str(int(v)) + '-' + mapping(v)).replace(' ', '_')

        print('Label:['+mapping(v)+']')
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
                # label_info_file.write('\nNumber of edges: ' + str(len(edges)) + ' for node: ' + mapping(v) + '[' + str(v) + ']' + '\n')
                if ignore_deg is not None and len(edges) > ignore_deg:
                    label_info_file.write('Ignoring: [' + mapping(v) + '] \t\t\t degree: [' +str(len(edges))+']\n')
                    continue
                for uv_tuple in edges:
                    edge = tuple(sorted(uv_tuple))
                    sub_g.add_edge(edge[0], edge[1], weight=E[edge])
                    bfs_q.put(uv_tuple[1])
            else:
                continue

        # relabel the nodes to reflect textual label
        nx.relabel_nodes(sub_g, mapping, copy=False)

        label_info_file.write(str('\n'))
        # Writing some statistics about the subgraph
        label_info_file.write(str(nx.info(sub_g)) + '\n')
        label_info_file.write('density: ' + str(nx.density(sub_g)) + '\n')
        label_info_file.write('list of the frequency of each degree value [degree_histogram]: ' + str(nx.degree_histogram(sub_g)) + '\n')
        # TODO: Add other statistics for better understanding of the subgraph.
        subg_edgelist = nx.generate_edgelist(sub_g, label_graph_el)
        label_info_file.close()
        nx.write_graphml(sub_g, label_graph_filepath)

        subgraph_lists.append(sub_g)

        print('Graph generated at: ' + label_graph_filepath)

    return subgraph_lists

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

def get_dataset_path():
    import platform
    if platform.system() == 'Windows':
        dataset_path = 'D:\Datasets\Extreme Classification'
    elif platform.system() == 'Linux':
        dataset_path = '/home/cs16resch01001/datasets/Extreme Classification'
    else: # OS X returns name "Darwin"
        dataset_path = '/Users/monojitdey/Downloads'
    print(platform.system(), "os detected.")

    return dataset_path

# dataset_path = get_dataset_path()

def main(args):
    test_file = os.path.join(args.dataset_path,args.dataset_name,args.test_file)
    label_map_file = os.path.join(args.dataset_path,args.dataset_name,args.label_map_file)
    print("Parameters:",args)
    label_graph = get_subgraph(test_file,label_map_file,level=args.level,subgraph_count=args.subgraph_count,ignore_deg=args.ignore_deg,root_node=args.node_id)

if __name__ == '__main__':
    # sample call: python utils/util.py /Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt /Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt
    # dataset_path = 'D:\Datasets\Extreme Classification'
    # dataset_name = 'Wiki10-31K'
    # test_file = 'Wiki10/wiki10_test.txt'
    # label_map_file = 'Wiki10-31K_mappings/wiki10-31K_label_map.txt'
    # Examples:
    # 1. python utils/util.py \n
    # 2. python utils/util.py --node_id 4844 \n
    # 3. python utils/util.py --test_file /Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt --label_map_file /Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt \n
    # 4. python utils/util.py --dataset_path /Users/monojitdey/Downloads/ --dataset_name Wiki10-31K --test_file /Wiki10/wiki10_test.txt --label_map_file /Wiki10-31K_mappings/wiki10-31K_label_map.txt \n
    # 5. python utils/util.py --dataset_path /Users/monojitdey/Downloads/ --dataset_name Wiki10-31K --test_file /Wiki10/wiki10_test.txt --label_map_file /Wiki10-31K_mappings/wiki10-31K_label_map.txt --node_id 4844 \n

    parser = ArgumentParser("Label Graph generator",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve',
                          epilog="Example: python utils/util.py --dataset_path /Users/monojitdey/Downloads/ --dataset_name Wiki10-31K --test_file /Wiki10/wiki10_test.txt --label_map_file /Wiki10-31K_mappings/wiki10-31K_label_map.txt --node_id 4844 \n")
    parser.add_argument('--dataset_path',
                        help='Path to dataset folder', type=str,
                        default=get_dataset_path())
    parser.add_argument('--dataset_name',
                        help='Name of the dataset to use', type=str,
                        default='Wiki10-31K')
    parser.add_argument('--test_file',# required=True,
                        help='Test file path inside dataset', type=str,
                        default='Wiki10/wiki10_test.txt')
    parser.add_argument('--label_map_file',
                        help="Label file path inside dataset. (If label file is not provided, graph will show numeric labels only)", type=str,
                        default='Wiki10-31K_mappings/wiki10-31K_label_map.txt')
    parser.add_argument('--level',
                        help='Number of hops to generate graph', type=int,
                        default=1)
    parser.add_argument('--ignore_deg',
                        help='Ignores nodes with degree >= [ignore_degree]', type=int,
                        default=500)
    parser.add_argument('--node_id',
                        help='ID [Row number on  file] of the root node to generate graph', type=int,
                        default=None)
    parser.add_argument('--subgraph_count',
                        help='How many subgraphs should be generated in single run', type=int,
                        default=5)
    args = parser.parse_args()

    main(args)
