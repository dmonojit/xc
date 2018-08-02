#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Tools for handling extreme classification datasets
__description__ :
__project__     : Extreme Classification
__author__      : Monojit Dey, Samujjwal Ghosh
__version__     :
__date__        : June 2018
__copyright__   : "Copyright (c) 2018"
__license__     : "Python"; (Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html)

__classes__     :

__variables__   :

__methods__     :

TODO            : 1. Claculate and normalize edge weights; Ew() = Edge weight.
                    Undirected: P(u,v) = Ew(u,v) / [Sum {Ew(u,x)} + Sum {Ew(v,x)}]
                    Directed  : P(u|v) = Ew(u,v) / Sum {Ew(u)} (alternatively, v can also be used in denominator)
TODO            : 2. Run Deepwalk, node2vec, LINE on whole graph and find k similar labels using embeddings.
"""

import sys,os,json
import networkx as nx
# import datetime
# import operator
# from functools import reduce
# from scipy.sparse import *
from scipy import *
from sklearn.datasets import load_svmlight_file
import itertools
import numpy as np
from queue import Queue  # Python 2.7 does not have this library
from collections import defaultdict

from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from unidecode import unidecode


def remove_special_chars(text,specials="""< >  * ? " / \ : |""",replace=' '):
    """
    Replaces [specials] chars from [text] with [replace]
    :param text:
    :param specials:
    :param replace:
    :return:
    """
    text = unidecode(str(text))
    trans_dict = {chars:replace for chars in specials}
    trans_table = str.maketrans(trans_dict)
    return text.translate(trans_table)


def get_dataset_path():
    """
    Returns dataset path based on OS.
    :return:
    """
    import platform

    if platform.system() == 'Windows':
        dataset_path = 'D:\Datasets\Extreme Classification'
        sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research')
    elif platform.system() == 'Linux':
        dataset_path = '/home/cs16resch01001/datasets/Extreme Classification'
        sys.path.append('/home/cs16resch01001/codes')
    else:  # OS X returns name 'Darwin'
        dataset_path = '/Users/monojitdey/ml/datasets'
    print(platform.system(),"os detected.")

    return dataset_path


def save_json(data,filename,file_path='',overwrite=False,indent=2,date_time_tag=''):
    """

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :param indent:
    :param date_time_tag:
    :return:
    """
    import json
    print("Saving JSON file: ", os.path.join(file_path,date_time_tag+filename+".json"))
    if not overwrite and os.path.exists(os.path.join(file_path,date_time_tag+filename+".json")):
        print("File already exists and Overwrite == False.")
        return True
    try:
        with open(os.path.join(file_path,date_time_tag+filename+".json"),'w') as json_file:
            try:
                json_file.write(json.dumps(data, indent=indent))
            except Exception as e:
                print("Writing json as string:",os.path.join(file_path,date_time_tag+filename+".json"))
                json_file.write(json.dumps(str(data), indent=indent))
                return True
        json_file.close()
        return True
    except Exception as e:
        print("Could not write to json file:",os.path.join(file_path,filename))
        print("Failure reason:",e)
        print("Writing file as plain text:",filename+".txt")
        write_file(data,filename,date_time_tag=date_time_tag)
        return False


from collections import OrderedDict
def load_json(filename,file_path='',overwrite=False,date_time_tag=''):
    # print("Reading JSON file: ",os.path.join(file_path,date_time_tag+filename+".json"))
    if os.path.isfile(os.path.join(file_path,date_time_tag+filename+".json")):
        with open(os.path.join(file_path,date_time_tag+filename+".json"), encoding="utf-8") as file:
            json_dict = OrderedDict(json.load(file))
        file.close()
        return json_dict
    else:
        print("Warning: Could not open file:",os.path.join(file_path,date_time_tag+filename+".json"))
        return False


def write_file(data,filename,file_path='',overwrite=False,mode='w',date_time_tag=''):
    """

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :param mode:
    :param date_time_tag:
    :return:
    """
    if not overwrite and os.path.exists(os.path.join(file_path,date_time_tag+filename+".txt")):
        print("File already exists and Overwrite == False.")
        return True
    with open(os.path.join(file_path,date_time_tag+filename+".txt"),mode,encoding="utf-8") as text_file:
        print("Saving text file: ", os.path.join(file_path,date_time_tag+filename+".txt"))
        text_file.write(str(data))
        text_file.write("\n")
        text_file.write("\n")
    text_file.close()


def get_x_and_y(filepath):
    """

    :param filepath:
    :return:
    """
    with open(filepath,'r') as file:
        total_points,feature_dm,number_of_labels = map(lambda x:int(x),file.readline().split(' '))

    X,Y = load_svmlight_file(filepath,n_features=feature_dm,multilabel=True,offset=1)

    return total_points,feature_dm,number_of_labels,X,Y


def get_matrices_from_file(filepath,label_filepath):
    """

    :param filepath:
    :param label_filepath:
    :return:
    """
    with open(filepath,'r') as file:
        total_points,feature_dm,number_of_labels = map(lambda x:int(x),file.readline().split(' '))

    total_points,feature_dm,number_of_labels,X,Y = get_x_and_y(filepath)

    label_graph = build_label_graph(Y,label_filepath)

    return total_points,feature_dm,number_of_labels,X,Y,label_graph


def get_x_y_v_e(filepath):
    """

    :param filepath:
    :return:
    """
    total_points,feature_dm,number_of_labels,X,Y = get_x_and_y(filepath)

    list_of_edge_lists = list(map(lambda x:list(itertools.combinations(x,2)),Y))

    V = list(set(itertools.chain.from_iterable(Y)))
    # E = set(itertools.chain.from_iterable(list_of_edge_lists))
    E = defaultdict(lambda:0)
    for e in itertools.chain.from_iterable(list_of_edge_lists):
        # sorted_edge = tuple(sorted(e))
        E[e] += 1

    return total_points,feature_dm,number_of_labels,X,Y,V,E


def find_single_labels(Y):
    """
    Finds the number of datapoints with only single label.
    """
    single_labels = []
    for i,t in enumerate(Y):
        if len(t) == 1:
            single_labels.append(i)
    if single_labels:
        print(len(single_labels),'datapoints has only single label. These labels will not occur in the co-occurrence graph.')
    return len(single_labels)


def get_label_dict(label_filepath):
    """

    :param label_filepath:
    :return:
    """
    if label_filepath is None:
        return {}

    try:
        with open(label_filepath,'r') as file:
            content = file.read().splitlines()
    except:
        with open(label_filepath,'r',encoding='latin-1') as file:  # Why 'latin-1' encoding?
            content = file.read().splitlines()

    label_dict = {}
    for i,label in enumerate(content):
        label_dict[i] = str(label)

    return label_dict


def get_subgraph(V,E,label_filepath,dataset_name,level=1,subgraph_count=5,ignore_deg=None,root_node=None):
    """
    # total_points: total number of data points
    # feature_dm: number of features per datapoint
    # number_of_labels: total number of labels
    # X: feature matrix of dimension total_points * feature_dm
    # Y: list of size total_points. Each element of the list containing labels corresponding to one datapoint
    # V: list of all labels (nodes)
    # E: dict of edge tuple -> weight, eg. {(1, 4): 1, (2, 7): 3}
    """

    # get a dict of label -> textual_label
    label_dict = get_label_dict(label_filepath)

    # an utility function to relabel nodes of upcoming graph with textual label names
    def mapping(v):
        """
        An utility function to relabel nodes of upcoming graph with textual label names
        :param v: label id (int)
        :return: returns the texual label of the node id [v]
        """
        v = int(v)
        if v in label_dict:
            return label_dict[v]
        return str(v)

    # build a unweighted graph of all edges
    g = nx.Graph()
    g.add_edges_from(E.keys())

    # Below section will try to build a smaller subgraph from the actual graph for visualization
    subgraph_lists = []
    for sg in range(subgraph_count):
        if root_node is None:
            # select a random vertex to be the root
            np.random.shuffle(V)
            v = V[0]
        else:
            v = root_node

        # two files to write the graph and label information
        # Remove characters like \, /, <, >, :, *, |, ", ? from file names,
        # windows can not have file name with these characters
        label_info_filepath = 'samples/'+str(dataset_name)+'_Info[{}].txt'.format(str(int(v)) + '-' + remove_special_chars(mapping(v)))
        label_graph_filepath = 'samples/'+str(dataset_name)+'_G[{}].graphml'.format(str(int(v)) + '-' + remove_special_chars(mapping(v)))
        # label_graph_el = 'samples/'+str(dataset_name)+'_E[{}].el'.format(str(int(v)) + '-' + mapping(v)).replace(' ','_')

        print('Label:[' + mapping(v) + ']')
        label_info_file = open(label_info_filepath,'w')
        label_info_file.write('Label:[' + mapping(v) + ']' + "\n")

        # build the subgraph using bfs
        bfs_q = Queue()
        bfs_q.put(v)
        bfs_q.put(0)
        node_check = {}
        ignored = []

        sub_g = nx.Graph()
        lvl = 0
        while not bfs_q.empty() and lvl <= level:
            v = bfs_q.get()
            if v == 0:
                lvl += 1
                bfs_q.put(0)
                continue
            elif node_check.get(v,True):
                node_check[v] = False
                edges = list(g.edges(v))
                # label_info_file.write('\nNumber of edges: ' + str(len(edges)) + ' for node: ' + mapping(v) + '[' +
                # str(v) + ']' + '\n')
                if ignore_deg is not None and len(edges) > ignore_deg:
                    # label_info_file.write('Ignoring: [' + mapping(v) + '] \t\t\t degree: [' + str(len(edges)) + ']\n')
                    ignored.append("Ignoring: deg [" + mapping(v) + "] = [" + str(len(edges)) + "]\n")
                    continue
                for uv_tuple in edges:
                    edge = tuple(sorted(uv_tuple))
                    sub_g.add_edge(edge[0],edge[1],weight=E[edge])
                    bfs_q.put(uv_tuple[1])
            else:
                continue

        # relabel the nodes to reflect textual label
        nx.relabel_nodes(sub_g,mapping,copy=False)
        print('sub_g:',sub_g)

        label_info_file.write(str('\n'))
        # Writing some statistics about the subgraph
        label_info_file.write(str(nx.info(sub_g)) + '\n')
        label_info_file.write('density: ' + str(nx.density(sub_g)) + '\n')
        label_info_file.write('list of the frequency of each degree value [degree_histogram]: ' +
                              str(nx.degree_histogram(sub_g)) + '\n')
        for nodes in ignored:
            label_info_file.write(str(nodes) + '\n')
        # TODO: Add other statistics for better understanding of the subgraph.
        # subg_edgelist = nx.generate_edgelist(sub_g,label_graph_el)
        label_info_file.close()
        nx.write_graphml(sub_g,label_graph_filepath)

        subgraph_lists.append(sub_g)

        print('Graph generated at: ' + label_graph_filepath)

        if root_node:
            print("Root node provided, will generate only one graph file.")
            break

    return subgraph_lists


def plot_occurance(E,plot_name='co-occurance.jpg',clear=True,log=False):
    from matplotlib import pyplot as plt
    plt.plot(E)
    plt.xlabel("Edges")
    if log:
        plt.yscale('log')
    plt.ylabel("Label co-occurance")
    plt.title("Label co-occurance counts")
    plt.savefig(plot_name)
    if clear:
        plt.cla()


def edge_stats(E):
    """
    Generates and returns edge related statistics about graph
    """
    import statistics as st  # Python 2.7 does not have this library

    e_stat = {}
    edge_occurances = E.values()
    edge_occurances_sorted = sorted(list(edge_occurances))
    # print(mean(edge_occurances_sorted))
    # print(median(edge_occurances_sorted))
    # print(st.mode(edge_occurances_sorted))
    # print(st.harmonic_mean(edge_occurances_sorted))
    # print(st.median_low(edge_occurances_sorted))
    # print(st.median_high(edge_occurances_sorted))
    # print(st.median_grouped(edge_occurances_sorted))

    e_stat['mean'] = mean(edge_occurances_sorted)
    e_stat['median'] = median(edge_occurances_sorted)
    e_stat['mode'] = st.mode(edge_occurances_sorted)
    # e_stat['harmonic_mean'] = st.harmonic_mean(edge_occurances_sorted)
    # e_stat['median_low'] = st.median_low(edge_occurances_sorted)
    # e_stat['median_high'] = st.median_high(edge_occurances_sorted)
    # e_stat['median_grouped'] = st.median_grouped(edge_occurances_sorted)

    # e_stat['edge_occurances_sorted'] = edge_occurances_sorted
    e_stat['edge_count'] = len(edge_occurances_sorted)
    
    from collections import Counter
    e_stat['edge_counter'] = Counter(edge_occurances_sorted)

    return e_stat,edge_occurances_sorted


def main(args):
    """

    :param args:
    :return:
    """

    datasets = ['RCV1-2K']
    for dataset in datasets:
        train_graph_file = dataset+'_train.txt'
        # train_graph_file = dataset+'/'+dataset+'_train.txt'
        train_graph_file = os.path.join(args.dataset_path,dataset,train_graph_file)

        # label_map = dataset+'_mappings/'+dataset+'_label_map.txt'
        # label_map_file = os.path.join(args.dataset_path,dataset,label_map)

        total_points,feature_dm,number_of_labels,X,Y,V,E = get_x_y_v_e(train_graph_file)

        save_json(V,dataset+'_V_train',os.path.join(args.dataset_path,dataset))
        save_json(E,dataset+'_E_train',os.path.join(args.dataset_path,dataset),overwrite=True)

        # Collecting some stats about the dataset and graph.
        e_stats,edge_occurances_sorted = edge_stats(E)
        e_stats['singles_train'] = find_single_labels(Y)
        save_json(e_stats,dataset+"_edge_statistics_train")

        plot_occurance(edge_occurances_sorted,plot_name=dataset+'_train_edge_occurances_sorted.jpg',clear=False)
        plot_occurance(edge_occurances_sorted,plot_name=dataset+'_train_edge_occurances_sorted_log.jpg',log=True)


        test_graph_file = dataset+'_test.txt'
        # test_graph_file = dataset+'/'+dataset+'_test.txt'
        test_graph_file = os.path.join(args.dataset_path,dataset,test_graph_file)

        # label_map = dataset+'_mappings/'+dataset+'_label_map.txt'
        # label_map_file = os.path.join(args.dataset_path,dataset,label_map)

        total_points,feature_dm,number_of_labels,X,Y,V,E = get_x_y_v_e(test_graph_file)

        save_json(V,dataset+'_V_test',os.path.join(args.dataset_path,dataset))
        save_json(E,dataset+'_E_test',os.path.join(args.dataset_path,dataset),overwrite=True)

        # Collecting some stats about the dataset and graph.
        e_stats,edge_occurances_sorted = edge_stats(E)
        e_stats['singles_test'] = find_single_labels(Y)
        save_json(e_stats,dataset+"_edge_statistics_test")

        plot_occurance(edge_occurances_sorted,plot_name=dataset+'_test_edge_occurances_sorted.jpg',clear=False)
        plot_occurance(edge_occurances_sorted,plot_name=dataset+'_test_edge_occurances_sorted_log.jpg',log=True)

    # label_graph_lists = get_subgraph(V,E,label_map_file,dataset_name=dataset,level=args.level,subgraph_count=args.subgraph_count,ignore_deg=args.ignore_deg,root_node=args.node_id)
    return


if __name__ == '__main__':
    # text = "Ceñía Lo+=r?e~~m ipsum dol;or sit!! amet, consectet..ur ad%"
    # print(remove_special_chars(text))
    # exit(0)
    """
    sample call: python utils/util.py /Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt
    /Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt dataset_path =
    'D:\Datasets\Extreme Classification' dataset_name = 'Wiki10-31K' test_file = 'Wiki10/wiki10_test.txt'
    label_map_file = 'Wiki10-31K_mappings/wiki10-31K_label_map.txt'

    Examples:
      1. python utils/util.py

      2. python utils/util.py --node_id 4844

      3. python utils/util.py --test_file /Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt --label_map_file
      /Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt

      4. python utils/util.py --dataset_path /Users/monojitdey/Downloads/ --dataset_name Wiki10-31K --test_file
      /Wiki10/wiki10_test.txt --label_map_file /Wiki10-31K_mappings/wiki10-31K_label_map.txt
      5. python utils/util.py --dataset_path /Users/monojitdey/Downloads/ --dataset_name Wiki10-31K --test_file
      /Wiki10/wiki10_test.txt --label_map_file /Wiki10-31K_mappings/wiki10-31K_label_map.txt --node_id 4844
    """
    parser = ArgumentParser("Label Graph generator",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            epilog="Example: python utils/util.py --dataset_path /Users/monojitdey/Downloads/ "
                                   "--dataset_name Wiki10-31K --test_file /Wiki10/wiki10_test.txt --label_map_file "
                                   "/Wiki10-31K_mappings/wiki10-31K_label_map.txt --node_id 4844 \n")
    parser.add_argument('--dataset_path',
                        help='Path to dataset folder',type=str,
                        default=get_dataset_path())
    parser.add_argument('--dataset_name',
                        help='Name of the dataset to use',type=str,
                        default='all')
    # parser.add_argument('--graph_file',  # required=True,
    #                     help='File path from which graph to be generated',type=str,
    #                     default='AmazonCat-13K/AmazonCat-13K_train.txt')
    # parser.add_argument('--label_map_file',
    #                     help="Label_map file path inside dataset. (If label file is not provided, graph will show "
    #                          "numeric labels only)",
    #                     type=str,
    #                     default='AmazonCat-13K_mappings/AmazonCat-13K_label_map.txt')
    parser.add_argument('--level',
                        help='Number of hops to generate graph',type=int,
                        default=1)
    parser.add_argument('--ignore_deg',
                        help='Ignores nodes with degree >= [ignore_degree]',type=int,
                        default=500)
    parser.add_argument('--node_id',
                        help='ID [Row number on  file] of the root node to generate graph',type=int,
                        default=12854)
    parser.add_argument('--subgraph_count',
                        help='How many subgraphs should be generated in single run',type=int,
                        default=1)
    args = parser.parse_args()
    # Dataset_names: Wiki10-31K AmazonCat-13K AmazonCat-14K Wikipedia-500K Amazon-670K Amazon-3M

    print("Parameters:",args)
    main(args)
