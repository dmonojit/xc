#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Code for DXML paper
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

TODO            : 1. Handle missing label from label co-occurrence graph
                    a. Count number of single labels for each dataset
                    b. Generate random vecs
                    c. remove those items from datasets.
TODO            : 2. Calculate edge weights as combination of co-occurrence an symantec similarity.
                    a. ConceptNet
                    b. Word2Vec
TODO            : 3. Run DXML with edge weights.
"""

import os,sys,logging
import numpy as np
import pickle as pk
from numpy.random import seed
seed(1)
from keras.models import Model,load_model
from keras.layers import Dense,Dropout,Input,Merge
from keras import regularizers
from keras import optimizers
import keras.backend as K
from tensorflow import set_random_seed
set_random_seed(2)
# from sklearn.preprocessing import MultiLabelBinarizer

from deepwalk import DeepWalk
from utils.util import get_x_y_v_e, get_dataset_path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def cal_loss(f_x,f_y):
    """
    Calculates loss function according to DXML.
    :param f_x: Feature vector of float or int
    :param f_y: Mean label vector of float or int
    :return:
    """
    return K.sum(K.tf.where(K.tf.less_equal(K.abs(f_x - f_y),1.),(K.pow((f_x - f_y),2.) / 2.,K.abs(f_x - f_y) - 0.5)))


class DXML(object):
    """
    Class for DXML neural netrowk architecture.
    """
    def __init__(self,model_save_path):
        # self.embedding = embedding
        self.model_save_path = model_save_path
        self.dxml = None

    def load_model(self,model_load_path=None):
        if not model_load_path:
            model_load_path = self.model_save_path

        self.dxml = load_model(model_load_path)
        print("Model succesfully loaded and compiled")

    def build(self,
              vec_len=300,
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
        X = Input(shape=(vec_len,),dtype='float64')
        F_y = Input(shape=(vec_len,),dtype='float64')

        if small:
            neuron = 256
            vec_len = 100

        print('Building model...')
        reg_coef = weight_decay / 2.0  # l2 = weight_decay / 2.0 by https://bbabenko.github.io/weight-decay/

        # DXML architecture with 2 fully-connected (Dense) layer with dropout at end.
        dxml_layer_1 = Dense(neuron,input_dim=X.shape,kernel_regularizer=regularizers.l2(reg_coef))
        dxml_layer_2 = Dense(vec_len,activation='relu',kernel_regularizer=regularizers.l2(reg_coef))(
            dxml_layer_1)  ## [neuron] value be equal to [vec_len] as |F_x| and |F_y| should be same dimension
        dxml_layer_do = Dropout(dropout)(dxml_layer_2)

        ## Taking feature output [F_x] after two layers as input with F_y
        F_x = dxml_layer_do(X)

        # Adding custom loss function as Merge layer
        dxml_distance = Merge(mode=cal_loss,output_shape=lambda x:(x[0][0],1))([F_x])
        self.dxml = Model([F_x,F_y],[dxml_distance])

        # SGD optimizer, with momentum
        optimizer = optimizers.SGD(lr,momentum)

        print(self.dxml.summary)
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


def gen_dw_vecs(E,dw_output_file='dw_output_file'):
    """
    Generates DeepWalk vectors
    :param E: list of edges for DeepWalk
    :param dw_output_file: file path to store generated vectors
    :return: vectors generated by DeepWalk
    """
    if os.path.exists(dw_output_file):  # checks if vectors were generated previously
        print('DeepWalk vectors file already exists at:',dw_output_file)
        from gensim.models import KeyedVectors

        dw_vecs = KeyedVectors.load_word2vec_format(dw_output_file)
        return dw_vecs

    dw_vecs = DeepWalk().transform(E.keys(),'edgelist')

    dw_vecs.wv.save_word2vec_format(dw_output_file)
    # dw_vecs.save(dw_output_file) # saves in non-human readable format
    print('Saved DeepWalk vectors at:',dw_output_file)
    return dw_vecs.wv


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


def fetch_avg_label_vecs(Y,dw_vecs,vec_len=64):
    """
    Takes mean of all the label vectors as per DXML.
    :param vec_len: Length of each label vectors
    :param Y: list of labels
    :param dw_vecs:
    :return:
    """
    print('fetch_avg_label_vecs')
    mean_lbl_vecs = {}
    for i,y in enumerate(Y):
        emb_lbl_vecs = np.zeros(shape=(1,vec_len),dtype=float)
        if len(y) <= 1:  # if length 1, no need to compute avg
            continue
        for y_i in y:
            emb_lbl_vecs = np.add(emb_lbl_vecs,dw_vecs[int(y_i)])
        mean_lbl_vecs[i] = np.divide(emb_lbl_vecs,len(y))
    return mean_lbl_vecs


def _test_fetch_avg_label_vecs():
    """
    Test function for [fetch_avg_label_vecs]
    """
    Y = [(1,2,3)]
    v1 = np.array([[1,2,3,4,5,6]],np.int32)
    v2 = np.array([[1,2,3,4,5,6]],np.int32)
    v3 = np.array([[1,2,3,4,5,6]],np.int32)
    dw_vecs = {1:v1,2:v2,3:v3}
    output = [1,2,3,4,5,6,]

    print('calculated value:',fetch_avg_label_vecs(Y,dw_vecs))
    print('actual value:',output)


if __name__ == "__main__":
    dataset_path = get_dataset_path()
    dataset = 'RCV1-2K'
    train_file = 'RCV1-2K_train.txt'
    test_file = 'RCV1-2K_test.txt'

    print('Loading pregenerated data.')
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

    ## Generating DeepWalk vectors
    dw_vecs_tr = gen_dw_vecs(E_tr,dw_output_file=os.path.join(dataset_path,dataset,'dw_vecs_tr.txt'))
    # dw_vecs_ts = gen_dw_vecs(E_ts,dw_output_file=os.path.join(dataset_path,dataset,'dw_vecs_ts.txt'))

    # print(dw_vecs_tr.vectors[2])

    # print(dw_vecs_tr.vectors.values())
    # y_vecs = []
    # for l in range(len(V_tr)):
    #     y_vecs.append(dw_vecs_tr.vectors[l])
    # y_tr_mat = np.matrix(y_vecs)
    # print(len(y_vecs))
    # print(y_tr_mat.shape)

    ## Calculating mean of label vectors belonging to a single datapoint
    Y_tr_mean_emb = fetch_avg_label_vecs(Y_tr,dw_vecs_tr.vectors)
    # print("Y_tr_mean_emb",Y_tr_mean_emb)

    exit(0)
