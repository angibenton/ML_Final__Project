from models import KernelPegasos
from kernel import SimplePairsKernel
from kernel import SimpleSubgraphsKernel
from kernel import TFIDFPairsKernel
from kernel import TFIDFSubgraphsKernel 

import pickle 
import pandas as pd
import numpy as np

#------TEST DRIVER FILE-------

def load_examples_from_csv(examples_filename):
    data_df = pd.read_csv(examples_filename)
    y = data_df["class"].to_numpy()
    conll = data_df["CoNLL"].values.tolist()
    strings = data_df["tweet"].values.tolist()
    X = tuple(zip(strings, conll))
    return X, y

def print_acc(y, y_hat):
    tot = len(y)
    correct = 0
    for i in range(tot):
        if y[i] == y_hat[i]:
            correct = correct + 1
    acc = correct / tot
    print("Accuracy = " + str(acc) + "%, " + str(correct) + " out of " + str(tot))

def train(examples_filename, model_filename, kernelmatrix_filename, lmbda):
    print("Training...")
    #get the data
    X, y = load_examples_from_csv(examples_filename)
    #get the kernel matrix
    kernel_matrix_df = pd.read_csv(kernelmatrix_filename)
    kernel_matrix = kernel_matrix_df.to_numpy()
    mod = KernelPegasos(nexamples = len(X), lmbda = lmbda) 
    #train
    mod.fit(X=X, y=y, kernel_matrix = kernel_matrix)  
    #save the model
    pickle.dump(mod, open(model_filename, 'wb'))
    print("Percentage of training examples that become support vectors: ",  len(mod.support_vectors)/len(X))
    print("Done training")
    return

def test(examples_filename, model_filename, kernel):
    print("Testing...")
    #open model
    mod = pickle.load(open(model_filename, 'rb'))
    #get the test data
    X, y = load_examples_from_csv(examples_filename)
    #compute the kernel_matrix
    print("computing kernel matrix for SVs vs. test examples...")
    kernel_matrix = kernel.compute_kernel_matrix(X = mod.support_vectors, X_prime = X)
    #predict
    y_hat = mod.predict(X=X, kernel_matrix=kernel_matrix)
    print_acc(y, y_hat)
    #save predictions ? idk
    print("Done testing")


def test_precomputed_kernel(examples_filename, model_filename, kernelmatrix_filename):
    print("Testing...")
    #open model
    mod = pickle.load(open(model_filename, 'rb'))
    #get the test data
    X, y = load_examples_from_csv(examples_filename)
    #open the kernel matrix
    kernel_matrix_df = pd.read_csv(kernelmatrix_filename)
    kernel_matrix = kernel_matrix_df.to_numpy()
    #predict
    y_hat = mod.predict(X, kernel_matrix)
    print_acc(y, y_hat)
    #save predictions ? idk 
    print("Done testing")

def precompute_train_kernel_matrix(X_train, kernel, save_matrix_filename):
    #compute kernel matrix
    print("computing kernel matrix...")
    kernel_matrix = kernel.compute_kernel_matrix(X=X_train)
    print("done computing kernel matrix of shape", kernel_matrix.shape)
    #save kernel matrix
    print("saving kernel matrix as csv...")
    kernel_dataframe = pd.DataFrame(data = kernel_matrix.astype(float))
    kernel_dataframe.to_csv(save_matrix_filename, index = False)
    print("done saving kernel matrix in " + save_matrix_filename)


def precompute_test_kernel_matrix(X_test, model_filename, kernel, save_matrix_filename):
    #open model
    mod = pickle.load(open(model_filename, 'rb'))
    #open test examples
    #compute kernel matrix 
    print("computing kernel matrix...")
    kernel_matrix = kernel.compute_kernel_matrix(X = mod.support_vectors, X_prime = X_test)
    print("done computing kernel matrix of shape", kernel_matrix.shape)
    #save kernel matrix
    print("saving kernel matrix as csv...")
    kernel_dataframe = pd.DataFrame(data = kernel_matrix.astype(float))
    kernel_dataframe.to_csv(save_matrix_filename, index = False)
    print("done saving kernel matrix in " + save_matrix_filename)



#PROBLEM ONE
#we all use the same X_train
#@jack this works for p2, not p1. I can't find the difference between the 2 files
X_train, y_train = load_examples_from_csv('data/p1_train_parsed.csv')
print(y_train.shape)

sim_pair_k = SimplePairsKernel()
precompute_train_kernel_matrix(X_train = X_train, kernel = sim_pair_k, save_matrix_filename = "kernel_matrices/p1_train_pairs.csv")

sim_graph_k = SimpleSubgraphsKernel()
precompute_train_kernel_matrix(X_train = X_train, kernel = sim_graph_k, save_matrix_filename = "kernel_matrices/p1_train_subgraphs.csv")

tfidf_pair_k = TFIDFPairsKernel(X=X_train)
precompute_train_kernel_matrix(X_train = X_train, kernel = tfidf_pair_k, save_matrix_filename = "kernel_matrices/p1_train_pairs_tfidf.csv")

tfidf_graph_k = TFIDFSubgraphsKernel(X=X_train)
precompute_train_kernel_matrix(X_train = X_train, kernel = tfidf_graph_k, save_matrix_filename = "kernel_matrices/p1_train_subgraphs_tfidf.csv")