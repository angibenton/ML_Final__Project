""" 
HW4 - KernelPegasos implmentation by Angi Benton, abenton3
"""

import numpy as np


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        raise NotImplementedError()


    def predict(self, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class KernelPegasos(Model):

    def __init__(self, *, nexamples, lmbda):
        """
        Args:
            nexamples: size of example space
            lmbda: regularizer term (lambda)

        Sets:
            b: beta vector (related to alpha in dual formulation)
            t: current iteration
            kernel_degree: polynomial degree for kernel function
            support_vectors: array of support vectors
            labels_corresp_to_svs: training labels that correspond with support vectors
        """
        super().__init__(lmbda=lmbda)
        self.b = np.zeros(nexamples, dtype=int)
        self.t = 1
        self.support_vectors = []
        self.labels_corresp_to_svs = []


    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        n = len(y) #number of examples
       
        for j in range(n): #outer loop over examples 
            self.t += 1 #increment time step

            #get the update condition 
            multiplier = y[j] / (self.lmbda * (self.t - 1)) 
            kernel_sum = 0
            for i in range(n): #inner loop over examples 
                kernel_sum += self.b[i] * y[i] * kernel_matrix[i][j]
            update_condition = multiplier * kernel_sum < 1      
            #update 
            if update_condition:
                self.b[j] += 1

        #save support vectors based on self.b nonzero values 
        self.support_vectors = [] #reset the lists on each epoch to avoid duplicate SV's 
        self.labels_corresp_to_svs = [] 
        for j in range(n):
            if self.b[j] != 0:
                self.support_vectors.append(X[j])
                self.labels_corresp_to_svs.append(y[j])



    def predict(self, *, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        #Calculate alpha, which is determined by beta
        alpha = []
        for i in range (len(self.b)):
            if self.b[i] != 0: #consider only nonzero beta (=> nonzero alpha)
                alpha.append(self.b[i] / (self.lmbda * self.t))

        #Predict labels based on the support vectors and their corresponding alphas 
        sums = np.zeros(len(X))
        for i in range (len(self.support_vectors)):
            sums += alpha[i] * self.labels_corresp_to_svs[i] * kernel_matrix[i]
        y_hat = np.sign(sums)
        return y_hat

