""" 
Keep model implementations in here.
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
        self.support_vectors = None
        self.labels_corresp_to_svs = None


    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        num_examples = y.shape[0]

        for j in range(num_examples):
            if y[j] == 0: #change from {0,1} to {-1,1}
                y[j] = -1
            self.t += 1

            sum = 0
            for i in range(num_examples):#compute sum
                sum += self.b[i]*y[i]*kernel_matrix[i,j]

            decision = float(y[j])/(float(self.lmbda)*(float(self.t) - 1))*float(sum)
            if decision < 1:
                self.b[j] += 1
        

        self.support_vectors = list()
        self.labels_corresp_to_svs = np.zeros(num_examples)
            
        for i in range(self.b.shape[0]):
            if(self.b[i] != 0):
                self.support_vectors.append(X[i])
                self.labels_corresp_to_svs[i] = y[i]
        


    def predict(self, *, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        a = np.zeros(self.b.shape[0], dtype=float)
        yhat = np.zeros(len(X),dtype=int)

        for i in range(self.b.shape[0]): #Recover ALL values of a
            a[i] = self.b[i] * 1/(self.lmbda * (self.t - 1))

        for j in range(len(X)): #predict all examples
            #loop over nonzero entries
            sum = 0
            a_temp = a[np.nonzero(a)]
            label_temp = self.labels_corresp_to_svs[np.nonzero(self.labels_corresp_to_svs)]
            for i in range(len(self.support_vectors)):
                sum += a_temp[i]*label_temp[i]*kernel_matrix[i,j]

            if sum > 0:
                yhat[j] = 1
            else:
                yhat[j] = 0

        return yhat
