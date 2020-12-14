#Workflow to save misclassified examples 
#for problem 2, using TFIDFSubgraphsKernel

import driver_functions as fn
from kernel import TFIDFSubgraphsKernel 
import pickle
import numpy as np
import pandas as pd

#Train using precomputed training kernel matrix 
fn.train(examples_filename = "data/p2_train_parsed.csv",
         model_filename = "saved_models/p2_best_model.model",
         kernelmatrix_filename = "kernel_matrices/p2_train_subgraphs_matrix_tfidf.csv",
         lmbda = 1e-6)

#Test      
print("Testing...")
#open model
mod = pickle.load(open("saved_models/p2_best_model.model", 'rb'))
#get the test data
X_test, y_test = fn.load_examples_from_csv("data/p2_test_parsed.csv")
#compute the kernel_matrix
print("computing kernel matrix for SVs vs. test examples...")
tfidf_graph_kernel = TFIDFSubgraphsKernel(X=mod.support_vectors, X_prime = X_test)
kernel_matrix = tfidf_graph_kernel.compute_kernel_matrix(X = mod.support_vectors, X_prime = X_test)
#predict
y_hat = mod.predict(X=X_test, kernel_matrix=kernel_matrix)
fn.print_acc(y_test, y_hat)
print("Done testing")

#SAVE MISCLASSIFIED 
print("Saving mis-classified examples...")
miss_y_true = []
miss_y_pred = []
miss_X = []
for i in range (len(y_test)):
    if (y_test[i] != y_hat[i]):
        miss_y_true.append(y_test[i])
        miss_y_pred.append(y_hat[i])
        miss_X.append(X_test[i])

miss_df = pd.DataFrame({
    "tweet": [tweet for tweet, conll in miss_X],
    "conll": [conll for tweet, conll in miss_X],
    "true": miss_y_true,
    "predicted": miss_y_pred})

miss_df.to_csv("data/p2_mistakes_bestmodel.csv")
print("Saved mis-classified examples to data/p2_mistakes_bestmodel.csv")


        

