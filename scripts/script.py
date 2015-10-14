import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from time import time

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn import cross_validation
from sklearn.random_projection import GaussianRandomProjection


"""
n_samples, n_features = data.shape
n_digits = len(np.unique(labels))
"""

def pca_lda_svc(data, labels):
    print "doing PCA"
    t0 = time()
    pca = PCA(n_components=n_digits).fit(data)
    print "PCA took %d seconds" % (time() - t0)
    data_pca = pca.transform(data)

    t0 = time()
    lda = LDA().fit(data_pca, labels)
    print "LDA took %d seconds" % (time() - t0)
    data_pca_lda = lda.transform(data_pca)

    print "dimension of pca'd data:", data_pca.shape

    print "evaluating SVC"
    t0 = time()
    svc = svm.SVC()
    scores = cross_validation.cross_val_score(svc, data_pca , labels, cv=5)
    print "SVC training and evaluation took %d seconds" % (time() - t0)

    print "Scores for PCA(n dimensions, n=10) + LDA + SVC:"
    print scores



def gaussrp_svc():
    print "loading data..."
    train_raw = np.array(pd.read_csv("../input/train.csv"))
    total_num_samples = train_raw.shape[0]

    kf = cross_validation.KFold(total_num_samples, n_folds=5)

    scores = []

    for i, (train_keys, test_keys) in enumerate(kf):
        print "iteration %d" % i

        # select the k-fold data
        train = train_raw[train_keys]
        test = train_raw[test_keys]

        # extract the features and labels
        train_data = scale(np.array(train[:,1:], dtype=np.float_))
        train_labels = train[:,0]

        test_data = scale(np.array(test[:,1:], dtype=np.float_))
        test_labels = test[:,0]


        # train GRP
        print "doing gaussian random projection..."
        t0 = time()
        grp = GaussianRandomProjection(eps=0.9).fit(train_data)
        print "GRP took %d seconds" % (time() - t0)


        # apply the GRP transformation
        print "applying the GRP transformation..."
        train_data_grp = grp.transform(train_data)
        test_data_grp = grp.transform(test_data)

        print train_data_grp.shape, test_data_grp.shape

        print "evaluating SVC"
        t0 = time()
        svc = svm.SVC()
        #scores = cross_validation.cross_val_score(svc, data_grp , labels, cv=5)
        svc.fit(train_data_grp, train_labels)
        s = svc.score(test_data_grp, test_labels)
        print "SVC training and evaluation took %d seconds" % (time() - t0)
        print "score is",s
        scores.append(s)

    print "final scores:", scores
    print "mean score:", np.mean(scores)

    """
    print "doing gaussian random projection..."
    t0 = time()
    grp = GaussianRandomProjection(eps=0.5).fit(data)
    data_grp = grp.transform(data)
    print "GRP took %d seconds" % (time() - t0)

    print data_grp.shape

    print "evaluating SVC"
    t0 = time()
    svc = svm.SVC()
    scores = cross_validation.cross_val_score(svc, data_grp , labels, cv=5)
    print "SVC training and evaluation took %d seconds" % (time() - t0)

    print "Scores for GRP (default params) + SVC:"
    print scores
    """


def load_raw_data():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train_raw = pd.read_csv("../input/train.csv")
    test_raw  = pd.read_csv("../input/test.csv")

    return train_raw, test_raw


def get_training_data():
    print "loading data..."
    train_raw = pd.read_csv("../input/train.csv")

    # just use part of the training data for now
    #SLICE_SIZE = 1000
    #train = train_raw.iloc[0:SLICE_SIZE,:]
    train = train_raw

    data = scale(np.array(train.iloc[:,1:], dtype=np.float_))
    labels = train.iloc[:,0]
    return data, labels

if __name__=="__main__":
    #data, labels = get_training_data()
    gaussrp_svc()
