from models import KernelPegasos
from kernel import SimplePairsKernel
from kernel import SimpleSubgraphsKernel
from kernel import TFIDFPairsKernel
from kernel import TFIDFSubgraphsKernel 

import pandas as pd
import numpy as np

#------TEST DRIVER FILE-------

#kernel matrix p2_train_pairs_matrix.csv (made by SimplePairsKernel)
#TRAIN
#get the kernel matrix
kernel_matrix_train_df = pd.read_csv("data/p2_train_pairs_matrix.csv")
kernel_matrix_train = kernel_matrix_train_df.to_numpy()

#get the data
train_data_df = pd.read_csv("data/p2_train_parsed.csv")
y_train = train_data_df["class"].to_numpy()
X_train = train_data_df["CoNLL"].values.tolist()

#train model
mod = KernelPegasos(nexamples = len(X_train), lmbda = 1e-3) #higher lamba -> higher amount of SVs
mod.fit(X=X_train, y=y_train, kernel_matrix = kernel_matrix_train)
print("Percentage of training examples that become support vectors: ",  len(mod.support_vectors)/len(X_train))


#will need this for saving models if we want
#pickle.dump(mod, open("p2_pairs.model", 'wb'))
#mod = pickle.load(open("p2_pairs.model", 'rb'))

#TEST 
#get the data
data_test_df = pd.read_csv("data/p2_test_parsed.csv")
y_test = data_test_df["class"].to_numpy()
X_test = data_test_df["CoNLL"].values.tolist()

#get the kernel matrix
sp_kernel = SimplePairsKernel()
print("computing kernel matrix for SVs vs. test examples...")
kernel_matrix_test = sp_kernel.compute_kernel_matrix(X = mod.support_vectors, X_prime = X_test)
print("test kernel matrix:\n", kernel_matrix_test)

y_hat = mod.predict(X=X_test, kernel_matrix=kernel_matrix_test)
print("predictions: \n", y_hat)


