#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Code for DXML paper
__description__ :
__project__     : Extreme Classification
__author__      : Samujjwal Ghosh, Monojit Dey
__version__     :
__date__        : June 2018
__copyright__   : "Copyright (c) 2018"
__license__     : "Python"; (Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html)

__classes__     :

__variables__   :

__methods__     :

TODO            : 1. Handle missing label from label co-occurrence graph
                    a. Avg of top k symantically similar labels, if texual information available.
                    b. Generate random vecs for missing labels.
                    c. Remove those datapoints from dataset.
TODO            : 2. Calculate edge weights as combination of co-occurrence an symantec similarity.
                    a. ConceptNet
                    b. Word2Vec
TODO            : 3. Run DXML with edge weights
                    a. node2vec
                    b. HARP
"""

import os,sys,logging
from typing import Dict

import random
import numpy as np
import pickle as pk
from numpy.random import seed
seed(1)
from keras.models import Model,load_model
from keras.layers import Dense,Dropout,Input,Merge,merge,Embedding
from keras import regularizers
from keras import optimizers
import keras.backend as K
from tensorflow import set_random_seed
set_random_seed(2)
# from sklearn.preprocessing import MultiLabelBinarizer

from deepwalk import DeepWalk
from node2vec import Node2Vec
from utils.util import get_x_y_v_e, get_dataset_path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

VECTOR_LENGTH = 300  ## Length of vectors
CLIPNORM = 0.5


'''
def build_dxml(x_train,
               y_train,
               x_test,
               y_test,
               small=False,
               momentum=0.9,
               # l=300,
               neuron=512,
               dropout=0.2,
               lr=0.015,
               batch_size=32,
               epochs=1,
               weight_decay=0.0005
               ):
    print('Build model...')
    # feature_count = 0  #
    # maxlen = 100
    reg_coef = weight_decay / 2.0  # l2 = weight_decay * 2.0; https://bbabenko.github.io/weight-decay/

    if small:
        neuron = 256
        # l = 100
    dxml = Sequential()

    dxml.add(Dense(neuron,input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(reg_coef)))
    dxml.add(Dense(neuron,activation='relu',kernel_regularizer=regularizers.l2(reg_coef)))
    dxml.add(Dropout(dropout))

    dxml.summary()
    # 'categorical_crossentropy'
    dxml.compile(loss=cal_loss,optimizer=optimizers.SGD(lr,momentum),metrics=['accuracy'])
    dxml.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.4)
    scores = dxml.evaluate(x_test,y_test)
    print("\n%s: %.2f%%" % (dxml.metrics_names[1],scores[1] * 100))

    predictions = dxml.predict(x_test)
    return predictions
'''


from scipy import sparse
def load_npz(filename,file_path=''):
    """
    Loads numpy objects from npz files.
    :param filename:
    :param file_path:
    :return:
    """
    print("Reading NPZ file: ",os.path.join(file_path,filename + ".npz"))
    if os.path.isfile(os.path.join(file_path,filename + ".npz")):
        npz = sparse.load_npz(os.path.join(file_path,filename + ".npz"))
        return npz
    else:
        print("Warning: Could not open file: ",os.path.join(file_path,filename + ".npz"))
        return False


def save_npz(data,filename,file_path='',overwrite=True):
    """
    Saves numpy objects to file.
    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :return:
    """
    print("Saving NPZ file: ",os.path.join(file_path,filename + ".npz"))
    if not overwrite and os.path.exists(os.path.join(file_path,filename + ".npz")):
        print("File already exists and Overwrite == False.")
        return True
    try:
        sparse.save_npz(os.path.join(file_path,filename + ".npz"),data)
        return True
    except Exception as e:
        print("Could not write to npz file:",os.path.join(file_path,filename + ".npz"))
        print("Failure reason:",e)
        return False


def save_pickle(data,pkl_file_name,pkl_file_path,overwrite=False):
    """
    saves python object as pickle file
    :param data:
    :param pkl_file_name:
    :param pkl_file_path:
    :param overwrite:
    :return:
    """
    # print("Method: save_pickle(data, pkl_file, tag=False)")
    print("Writing to pickle file: ",os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
    if not overwrite and os.path.exists(os.path.join(pkl_file_path,pkl_file_name + ".pkl")):
        print("File already exists and Overwrite == False.")
        return True
    try:
        if os.path.isfile(os.path.join(pkl_file_path,pkl_file_name + ".pkl")):
            print("Overwriting on pickle file:",os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
        with open(os.path.join(pkl_file_path,pkl_file_name + ".pkl"),'wb') as pkl_file:
            pk.dump(data,pkl_file)
        pkl_file.close()
        return True
    except Exception as e:
        print("Could not write to pickle file:",os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
        print("Failure reason: ",e)
        return False


def load_pickle(pkl_file_name,pkl_file_path):
    """
    Loads pickle file from files.
    :param pkl_file_name:
    :param pkl_file_path:
    :return:
    """
    print("Method: load_pickle(pkl_file)")
    if os.path.exists(os.path.join(pkl_file_path,pkl_file_name + ".pkl")):
        print("Reading pickle file: ",os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
        with open(os.path.join(pkl_file_path,pkl_file_name + ".pkl"),'rb') as pkl_file:
            loaded = pk.load(pkl_file)
        return loaded
    else:
        print("Warning: Could not open file:",os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
        return False


def cal_loss(ip):
    """
    Calculates loss according to DXML.
    :param ip:
    :param f_x: Feature vector of float or int
    :param f_y: Average of label vectors of float or int
    :return:
    """
    f_x = ip[0]
    f_y = ip[1]
    # print(args)
    print(f_x.shape)
    print(f_x)
    K.print_tensor(f_x, message='f_x = ')
    print(f_y.shape)
    print(f_y)
    K.print_tensor(f_y, message='f_y = ')
    assert(f_x.shape[1] == f_y.shape[1])
    return K.sum(K.tf.where(K.tf.less_equal(K.abs(f_x - f_y),1.),(K.pow((f_x - f_y),2.) / 2.,K.abs(f_x - f_y) - 0.5)))


def custom_loss(l):
    def loss(f_x, f_y):
        return K.sum(K.tf.where(K.tf.less_equal(K.abs(f_x - f_y),1.),(K.pow((f_x - f_y),2.) / 2.,K.abs(f_x - f_y) - 0.5)))
    return loss


class DXML(object):
    """
    Class for DXML neural netrowk architecture.
    """
    def __init__(self,model_save_path):
        self.model_save_path = model_save_path
        self.dxml = None

    def load_model(self,model_load_path=None):
        if not model_load_path:
            model_load_path = self.model_save_path

        self.dxml = load_model(model_load_path)
        print("Model succesfully loaded.")

    def build(self,
              emb_vec_mat, ## Label embedding vector
              vec_len=VECTOR_LENGTH,
              max_features=00, ## Number of labels
              small=False,
              momentum=0.9,
              lr=0.015,
              neuron=512,
              dropout=0.2,
              weight_decay=0.0005
              ):
        """
        Builds the neural network given in DXML paper.
        :param vec_len: Length of input vectors
        :param small: True if dataset is small
        :param momentum:
        :param lr:
        :param neuron: Number of neurons to use
        :param dropout:
        :param weight_decay:
        """
        X   = Input(shape=(vec_len,),dtype='float32',name='X')
        F_y = Input(shape=(vec_len,),dtype='float32',name='F_y')

        # X_E = Embedding(max_features,vec_len,weights=[],input_length=1,trainable=True)(X)
        F_y_E = Embedding(max_features,vec_len,weights=[emb_vec_mat],input_length=1,trainable=False)(F_y)

        if small:
            neuron = 256
            vec_len = 100

        print('Building model...')
        reg_coef = weight_decay / 2.0  ## l2 = weight_decay / 2.0; by https://bbabenko.github.io/weight-decay/

        ## DXML architecture with 2 fully-connected (Dense) layer with dropout at end.
        dxml_layer_1 = Dense(neuron,input_dim=X.shape,kernel_regularizer=regularizers.l2(reg_coef))(X)
        dxml_layer_2 = Dense(vec_len,activation='relu',kernel_regularizer=regularizers.l2(reg_coef))(
            dxml_layer_1)  ## [neuron] value be equal to [vec_len] as |F_x| and |F_y| should be same dimension
        F_x = Dropout(dropout)(dxml_layer_2)

        ## Taking feature output [F_x] after two layers as input with F_y
        # F_x = dxml_layer_do(X)

        ## Adding custom loss function as Merge layer
        # dxml_distance = Merge(mode=cal_loss,output_shape=lambda x:(x[0][0],1))([F_x,F_y])
        dxml_distance = merge([F_x,F_y_E], mode=cal_loss, dot_axes=1, name="F_x_F_y_loss",output_shape=(vec_len,))
        self.dxml = Model(inputs=[X,F_y],outputs=[dxml_distance])

        print(self.dxml.summary)

        ## SGD optimizer, with momentum; [clipnorm] set the value for maximum possible gradient value
        optimizer = optimizers.SGD(lr=lr,momentum=momentum,clipnorm=CLIPNORM)

        # plot_model(self.dxml,to_file='DXML_Keras.png')
        self.dxml.compile(loss='mean_absolute_error',optimizer=optimizer,metrics=['accuracy'])

    def train(self,F_x_tr,F_y_tr,val_split=0.4,batch_size=64,num_epochs=5,save=True):
        self.dxml.fit([F_x_tr],F_y_tr,batch_size=batch_size,epochs=num_epochs,validation_split=val_split)

        if save:
            self.dxml.save(self.model_save_path)
            print("Model succesfully saved on disk at: %s" % self.model_save_path)

    def predict(self,X_ts,Y_ts):
        """
        Score the Y_ts w.r.t to a X_ts
        :param X_ts : single X_ts (string)
        :param Y_ts : list of candidate queries each a string
        :return : ranked list of Y_ts (candidate, similarity)
        """
        return self.dxml.predict([X_ts,Y_ts])


def gen_embs(E,emb_file_name='label_emb',emb_file_path='',nrl='deepwalk',format='edgedict',walks=20,
             vec_len=VECTOR_LENGTH,seed=0,walk_len=20,window=7,work=8,p=1,q=1,weight=False,directed=False,iter=1):
    """
    Generates DeepWalk vectors
    :param format: Input format of the graph; default: 'edgedict'
    :param nrl: Which NRL to use ['deepwalk', 'node2vec']
    :param directed: Flag to denote if graph is directed.
    :param weight: Flag to denote if graph is weighted.
    :param emb_file_path: Path to save generated embeddings
    :param work: Number of workers to use
    :param window: window size for skip-gram model
    :param walk_len: length of each random walk
    :param seed:
    :param vec_len: Length of generated vectors
    :param walks: Number of walks per node
    :param E: list of edges for DeepWalk in [edgelist] format.
    :param emb_file_name: file path to store generated vectors in w2v format.
    :return: Generated Embeddings in Gensim KeyedVector format.
    """

    label_emb_wv = None

    ## renaming output file with DeepWalk param values
    emb_file_name = emb_file_name+'_'+str(nrl)+'_'+str(walks)+'_'+str(vec_len)+'_'+str(walk_len)+'_'+str(window)\
                    +'_'+str(work)+'_'+str(weight)+'_'+str(directed)
    emb_file_name = os.path.join(emb_file_path,emb_file_name)
    
    if os.path.exists(emb_file_name):  ## checks if embeddings were generated previously
        
        print('Embeddings already exist at:',emb_file_name)
        from gensim.models import KeyedVectors

        label_emb_wv  = KeyedVectors.load_word2vec_format(emb_file_name)

    else:

        if nrl == 'deepwalk':
            ## DeepWalk default values: number_walks=10,representation_size=64,seed=0,walk_length=40,window_size=5,workers=1
            label_emb = DeepWalk(number_walks=walks,representation_size=vec_len,seed=seed,walk_length=walk_len,
                                 window_size=window,workers=work).transform(E,format)
        elif nrl == 'node2vec':
            ## Node2Vec default values: num_walks=10,dimensions=64,walk_length=40,window_size=5,workers=1,p=1,q=1,
            ## weighted=False,directed=False,iter=1
            label_emb = Node2Vec(num_walks=walks,dimensions=vec_len,walk_length=walk_len,window_size=window,workers=work,
                                 p=p,q=q,weighted=weight,directed=directed,iter=iter).transform(E,format)
        else:
            raise NotImplemented

        label_emb_wv = label_emb.wv

        directory = os.path.dirname(emb_file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        label_emb_wv.save_word2vec_format(emb_file_name)
        # label_emb.save(emb_file_name) # saves in non-human readable format
        print('Saved generated vectors at:',emb_file_name)

    # FYI
    # (pdb) label_emb_wv.__dict__.keys()
    # => dict_keys(['vectors', 'vocab', 'vector_size', 'index2word', 'vectors_norm'])
    
    return label_emb_wv


def _test_fetch_avg_label_vecs():
    """
    Test function for [avg_label_embs]
    """
    Y = [(1,2,3)]
    v1 = np.array([[1,2,3,4,5,6]],np.int32)
    v2 = np.array([[1,2,3,4,5,6]],np.int32)
    v3 = np.array([[1,2,3,4,5,6]],np.int32)
    dw_vecs = {1:v1,2:v2,3:v3}

    print('actual values:',[[1,2,3,4,5,6]])
    print('calculated values:',avg_label_embs(Y,dw_vecs))
    return True


def find_missing_labels(V,E):
    """
    Finds the missing label ids by taking set difference.
    :param V:
    :param E:
    :return: List of label indexes missing from edge list.
    """
    edge_list = set()
    for (u,v) in E.keys():
        edge_list.update((u,v))
    return list(set(V) - edge_list)


def gen_miss_vecs(labels,label_emb_wv,dim=VECTOR_LENGTH,low=-1.0,high=1.0):
    """
    Generates random vector uniform values for labels missing in DeepWalk vectors.
    :param high:
    :param low:
    :param dim: Dimension of the generated vector
    :param labels: list of missing labels
    :param label_emb_wv: word vector model of label embeddings
    """
    label_entities = []
    label_embs = []
    for m_label in labels:
        label_entities.append(str(int(m_label)))
        label_embs.append(np.random.uniform(low=low,high=high,size=(dim,)))
    
    label_emb_wv.add(label_entities, label_embs)


def avg_label_embs(Y,label_emb_wv,vec_len=VECTOR_LENGTH):
    """
    Takes mean of all the label vectors as per DXML.
    :param vec_len: Length of each label vectors
    :param Y: list of label indexes
    :param label_emb_wv: word vector model of label embeddings
    :return:
    """
    print('avg_label_embs',len(Y))
    # label_emb_mat = np.empty((0,vec_len))
    avg_lbl_embs = {}
    no_lbl       = 0  ## to count number of data points with no label
    single_lbl   = 0  ## to count number of data points with single label
    multi_lbl    = 0  ## to count number of data points with more than 1 label
    for i,y in enumerate(Y):
        emb_lbl_vecs = np.zeros(shape=(1,vec_len),dtype=float)
        if len(y) <= 1:  # if length 1, no need to compute avg
            if len(y) == 0:
                y_temp = random.choice([*label_emb_wv.vocab.keys()])
                no_lbl += 1
            else:
                y_temp = y[0]
                single_lbl += 1
            avg_lbl_embs[i] = np.reshape(label_emb_wv.get_vector(str(int(y_temp))), (-1, vec_len))
            # label_emb_mat = np.vstack((label_emb_mat,lbl_vecs[int(y[0])]))
            # if i <= 1:
                # print(avg_lbl_embs[i].shape)
            continue
        multi_lbl += 1
        for y_i in y:
            emb_lbl_vecs = np.add(emb_lbl_vecs,label_emb_wv.get_vector(str(int(y_i))))
        avg_lbl_embs[i] = np.divide(emb_lbl_vecs,len(y))
        # label_emb_mat = np.append(label_emb_mat,np.divide(emb_lbl_vecs,len(y)),axis=0)
    print('# data points with no label:',no_lbl)
    print('# data points with single label:',single_lbl)
    print('# data points with more than 1 label:',multi_lbl)
    print('Total # data points:',single_lbl + multi_lbl)
    return avg_lbl_embs#,label_emb_mat


if __name__ == "__main__":
    dataset_path = get_dataset_path()
    dataset = 'eurlex'

    train_file = dataset+'_train.txt'
    test_file  = dataset+'_test.txt'
    train_file = os.path.join(dataset_path,dataset,train_file)

    print('Loading pre-generated data.')
    X_tr = load_npz("X_tr",file_path=os.path.join(dataset_path,dataset))
    Y_tr = load_pickle(pkl_file_name="Y_tr",pkl_file_path=os.path.join(dataset_path,dataset))
    V_tr = load_pickle(pkl_file_name="V_tr",pkl_file_path=os.path.join(dataset_path,dataset))
    E_tr = load_pickle(pkl_file_name="E_tr",pkl_file_path=os.path.join(dataset_path,dataset))
    if not E_tr:
        print('Generating data from train and test files.')
        _,_,_,X_tr,Y_tr,V_tr,E_tr = get_x_y_v_e(os.path.join(dataset_path,dataset,train_file))
        # _,_,_,X_ts,Y_ts,V_ts,E_ts = get_x_y_v_e(os.path.join(dataset_path,dataset,test_file))

        save_npz(X_tr,"X_tr",file_path=os.path.join(dataset_path,dataset),overwrite=False)
        save_pickle(Y_tr,pkl_file_name="Y_tr",pkl_file_path=os.path.join(dataset_path,dataset))
        save_pickle(V_tr,pkl_file_name="V_tr",pkl_file_path=os.path.join(dataset_path,dataset))

        ## Converting [E_tr] to default dict as returned [E_tr] is not pickle serializable
        E_tr_2 = {}  ## TODO: Make returned [E_tr] pickle serializable
        for i,j in E_tr.items():
            E_tr_2[i] = j
        save_pickle(E_tr_2,pkl_file_name="E_tr",pkl_file_path=os.path.join(dataset_path,dataset))

    ## Creating One-Hot vectors from list of label indexes
    # from sklearn.preprocessing import MultiLabelBinarizer
    # mlb = MultiLabelBinarizer()
    # Y_tr_mlb = mlb.fit_transform(Y_tr)
    # Y_ts_mlb = mlb.fit_transform(Y_ts)

    ## Generating label embeddings using NRL

    # dw_vecs_tr = gen_embs(E_tr,emb_file_path=os.path.join(dataset_path,dataset),emb_file_name='V_tr_emb',nrl='deepwalk')

    label_emb_wv = gen_embs(E_tr,emb_file_path=os.path.join(dataset_path,dataset),emb_file_name='V_tr_emb',nrl='deepwalk')

    # dw_vecs_ts = gen_embs(E_ts,dw_file_path=os.path.join(dataset_path,dataset),dw_output_file='Y_ts_dw_vecs')
    # print(dir(dw_vecs_tr))

    print("Count of missing labels:",len(V_tr) - len(label_emb_wv.vectors))
    print(label_emb_wv.vectors.shape)
    missing_labels = []
    if len(V_tr) - len(label_emb_wv.vectors) > 0:
        missing_labels = find_missing_labels(V_tr,E_tr)
        assert(len(missing_labels) == len(V_tr) - len(label_emb_wv.vectors))
        gen_miss_vecs(missing_labels,label_emb_wv,label_emb_wv.vectors.shape[1])

    # y_vecs = []
    # for l in range(len(V_tr)):
    #     y_vecs.append(dw_vecs_tr.vectors[l])
    # y_tr_mat = np.matrix(y_vecs)
    # print(len(y_vecs))
    # print(y_tr_mat.shape)

    print(len(Y_tr))
    print(len(Y_tr[0]))
    print(Y_tr[0])

    ## Calculating mean of label vectors belonging to a single datapoint
    # Y_tr_mean_emb,label_emb_mat = avg_label_embs(Y_tr,dw_vecs_tr.vectors)
    Y_tr_mean_emb = avg_label_embs(Y_tr,label_emb_wv)

    ## Generating average label embedding matrix.
    print(len(Y_tr_mean_emb))
    label_emb_mat = np.empty((len(Y_tr_mean_emb.keys()),VECTOR_LENGTH))
    for k,v in Y_tr_mean_emb.items():
        label_emb_mat = np.vstack([label_emb_mat,v])

    print(label_emb_mat.shape)
    dxml_obj = DXML(os.path.join(dataset_path,dataset))
    dxml_obj.build(label_emb_mat,max_features=len(V_tr))
    dxml_obj.train()
