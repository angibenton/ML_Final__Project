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
train_conll = train_data_df["CoNLL"].values.tolist()
train_strings = train_data_df["tweet"].values.tolist()
X_train = tuple(zip(train_strings, train_conll))

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
test_conll = data_test_df["CoNLL"].values.tolist()
test_strings = data_test_df['tweet'].tolist()
X_test = tuple(zip(test_strings, test_conll))

#get the kernel matrix
sp_kernel = SimplePairsKernel()
print("computing kernel matrix for SVs vs. test examples...")
kernel_matrix_test = sp_kernel.compute_kernel_matrix(X = mod.support_vectors, X_prime = X_test)
print("test kernel matrix:\n", kernel_matrix_test)
df = pd.DataFrame(data = kernel_matrix_test.astype(float))

df.to_csv('./data/p2_test_pairs_matrix.csv', index = False)

y_hat = mod.predict(X=X_test, kernel_matrix=kernel_matrix_test)
print("predictions: \n", y_hat)

df2 = pd.DataFrame(data = y_hat.astype(float))

df2.to_csv('./data/p2_pairs_predictions.csv', index = False)


