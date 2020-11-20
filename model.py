import numpy as np
from numpy import array, dot
from qpsolvers import solve_qp
#currently working on this ! - angi 

#TESTING 
y = [1, 1, -1, -1]
kernel_matrix = [[10, 5, 1, 0],
                 [5, 10, 2, 3],
                 [1, 2, 10, 7],
                 [0, 3, 7, 10]]
x = ["friend", "love", "bitch", "fuck"]


class DualSVM(object):
    """ Implementation of the dual formulation of a support vector machine. 
        Relies on an external kernel matrix. 
    """

    def __init__(self, num_examples, margin_hardness):
        """ Dual SVM constructor.

        Args:
            num_examples: the number of examples that the model will be trained on 
            (in dual formulation this is also the number of learned parameters)
            margin_hardness: hyperparameter that represents how much we penalize misclassifications 
        """
        self.num = num_examples
        self.alphas = []
        self.nonzero_alphas = []
        self.support_vectors = []
        self.C = margin_hardness

    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model by solving the quadratic program that arises from the dual DVM w/ kernel trick. 

        Args:
            X: A list of self.num strings - do we need this other than for saving SVs ?
            y: an column vector of self.num binary labels, -1 or 1
            kernel_matrix: a symmetric, self.num x self.num matrix of precomputed kernels 
        """

        # Setup the quadratic program (negate the objective function because qpsolvers minimizes)
        q = (-1) * np.ones(self.num) # linear term 
        print("q: ", q)
        P = np.transpose(y) * kernel_matrix # quadratic term 
        print("P: ", P)

        # alpha dot y = 0 constraint 
        A = np.transpose(y)
        print("A: ", A)
        b = 0
        print("b: ", b)

        # all alphas between 0 and C constraint
        lb = np.zeros()
        print("lb: ", lb)
        ub = np.full((self.num, 1), self.C)
        print("ub: ", ub)

        # Solve (QP): 
        # MIN 1/2 alpha^t P alpha + q^t alpha
        # SUBJECT TO
        # A dot alpha = b
        # lb <= alpha <= ub
        self.alphas = solve_qp(P=P, q=q, A=A, b=b, lb=lb, ub=ub)
        print(self.alphas)

        # Save support vectors 
        for i in range(self.num):
            if (self.alphas[i] > 0):
                self.nonzero_alphas.append(self.alphas[i])
                self.support_vectors.append(X[i])


    def predict(self, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings to classify 
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        

    def get_support_vectors():
        return self.support_vectors 



