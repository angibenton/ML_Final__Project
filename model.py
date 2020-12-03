import numpy as np
from numpy import array, dot
from qpsolvers import solve_qp
#currently working on this ! - angi 




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
        self.support_vector_labels = []
        self.C = margin_hardness
        self.b = 0 #???????

    def fit(self, X, y, kernel_matrix):
        """ Fit the model by solving the quadratic program that arises from the dual DVM w/ kernel trick. 

        Args:
            X: A list of self.num examples - (tweet, tree) tuple? 
            y: an column vector of self.num binary labels, -1 or 1
            kernel_matrix: a symmetric, self.num x self.num matrix of precomputed kernels 
        """

        # Setup the quadratic program (negate the objective function because qpsolvers minimizes)
        q = (-1) * np.ones(self.num).astype(float) # linear term 
        print("q: ", q)

        P = (np.transpose(y) * kernel_matrix * y).astype(float) # quadratic term 
        print("P: ", P)
        print(np.linalg.eigvals(P))

        # alpha dot y = 0 constraint 
        A = np.transpose(y).astype(float)
        print("A: ", A)
        b = 0
        print("b: ", b)

        # all alphas between 0 and C constraint
        lb = np.zeros(self.num).astype(float)
        print("lb: ", lb)
        ub = np.full(self.num, self.C).astype(float)
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
                self.support_vector_labels.append(y[i])
        
        y_hat = np.zeros(len(X)) #initialize the predictions to zero
        for i in range(len(X)): #over examples (rows of kernel matrix)
            for j in range(len(self.support_vectors)): #over support vectors (columns of kernel matrix)
                y_hat[i] += self.nonzero_alphas[j] * self.support_vector_labels[j] * kernel_matrix[i][j]
        for i in range(len(y)):
            b = y[i] - y_hat[i] 
            print("Example #" +  str(i) + " - f(x): " + str(y_hat[i]) + ", y: " + str(y[i]) + ", b would be: " + str(b))


    def predict(self, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of examples to classify
            kernel_matrix: an ndarray containing kernel evaluations between the examples and support vectors 

        Returns:
            An array of ints with shape [num_examples].
        """
        y_hat = np.zeros(len(X)) #initialize the predictions to zero
        for i in range(len(X)): #over examples (rows of kernel matrix)
            for j in range(len(self.support_vectors)): #over support vectors (columns of kernel matrix)
                y_hat[i] += self.nonzero_alphas[j] * self.support_vector_labels[j] * kernel_matrix[i][j]
        y_hat += self.b
        y_hat = np.sign(y_hat)
        return y_hat

        

    def get_support_vectors(self):
        return self.support_vectors 




#TRAINING 
x = ["friend", "love", "bad word 3", "bad word 4"]
y = [1, 1, -1, -1]
kernel_matrix = [[100, 5, 1, 0],
                 [5, 100, 2, 3],
                 [1, 2, 100, 4],
                 [0, 3, 4, 100]]
print(np.linalg.eigvals(kernel_matrix))
mod = DualSVM(4, 100)
mod.fit(x, y, kernel_matrix)

#TESTING 2
x2 = ["bad word", "bad word 2"]
y2 = [-1, -1]
kernel_matrix2 = [[2, 0, 40, 8],
                 [1, 1, 20, 10]]
preds2 = mod.predict(x2, kernel_matrix2)
print("actual 2: ", end = '')
print(y2)
print("pred 2: ", end = '')
print(preds2)

#TESTING 3
x3 = ["happy", "cute"]
y3 = [1, 1]
kernel_matrix3 = [[100, 57, 1, 2],
                 [500, 350, 20, 10]]
preds3 = mod.predict(x3, kernel_matrix3)
print("actual 3: ", end = '')
print(y3)
print("pred 3: ", end = '')
print(preds3)









