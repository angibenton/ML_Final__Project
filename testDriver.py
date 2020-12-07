from models import KernelPegasos
from kernel import SimplePairsKernel
from kernel import SimpleSubgraphsKernel
from kernel import TFIDFPairsKernel
from kernel import TFIDFSubgraphsKernel 

import pickle 
import pandas as pd
import numpy as np

#------TEST DRIVER FILE-------

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
    data_df = pd.read_csv(examples_filename)
    y = data_df["class"].to_numpy()
    conll = data_df["CoNLL"].values.tolist()
    strings = data_df["tweet"].values.tolist()
    X = tuple(zip(strings, conll))
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
    data_df = pd.read_csv(examples_filename)
    y = data_df["class"].to_numpy()
    conll = data_df["CoNLL"].values.tolist()
    strings = data_df["tweet"].values.tolist()
    X = tuple(zip(strings, conll))
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
    data_df = pd.read_csv(examples_filename)
    y = data_df["class"].to_numpy()
    conll = data_df["CoNLL"].values.tolist()
    strings = data_df["tweet"].values.tolist()
    X = tuple(zip(strings, conll))
    #open the kernel matrix
    kernel_matrix_df = pd.read_csv(kernelmatrix_filename)
    kernel_matrix = kernel_matrix_df.to_numpy()
    #predict
    y_hat = mod.predict(X, kernel_matrix)
    print_acc(y, y_hat)
    #save predictions ? idk 
    print("Done testing")

#TODO add another function to just compute the testing kernel matrix so you can use this ^ if needed ?

#use different kernels here!
LMBDA = 1e-6
sp_kernel = SimplePairsKernel()
#NOTE: make lambda smaller -> less SVs -> less time to compute kernel matrix -> but at some point less accuracy it seems like
train(examples_filename = "data/p2_train_parsed.csv", model_filename = "saved_models/p2_simple_pairs.model", kernelmatrix_filename = "data/p2_train_pairs_matrix.csv", lmbda = LMBDA)
test(examples_filename = "data/p2_test_parsed.csv", model_filename = "saved_models/p2_simple_pairs.model", kernel = sp_kernel)



