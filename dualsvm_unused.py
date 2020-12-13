import numpy as np
from numpy import array, dot
from qpsolvers import solve_qp


'''
This model is not used in our final workflow because the type of kernel that we use 
does not meet the standard of a mathematical kernel, i.e. the dot product of two feature vectors.
Our kernels are, as those in hw4, more abstract notions of similarity between two tweets.
Still, this was an interesting pursuit so I would like to submit it with our used code. 
'''


class DualSVM(object):
    """ Implementation of the dual formulation of a support vector machine. 
        Interprets the dual formulation as a convex quadratic program. 
        Relies on an external kernel matrix that must be Positive Definite.
    """

    def __init__(self, num_examples, margin_hardness):
        """ Dual SVM constructor.

        Args:
            num_examples: the number of examples that the model will be trained on 
            (in dual formulation this is also the number of learned parameters)
            margin_hardness: hyperparameter that represents how much we penalize misclassifications 
        """
        self.num = num_examples
        self.alphas = [] #len = self.num
        self.nonzero_alphas = [] #len = num support vecs
        self.support_vectors = [] #len = num support vecs
        self.support_vector_labels = [] #len = num support vecs
        self.C = margin_hardness
        self.b = 0 

    def fit(self, X, y, kernel_matrix):
        """ Fit the model by solving the quadratic program that arises from the dual DVM w/ kernel trick. 

        Args:
            X: A list of self.num examples - (tweet, tree) tuples
            y: an column vector of self.num binary labels, -1 or 1
            kernel_matrix: a symmetric, self.num x self.num matrix of precomputed kernels 
        """

        # Setup the quadratic program (negate the objective function because qpsolvers minimizes)
        q = (-1) * np.ones(self.num).astype(float) # linear term 
        P = (np.transpose(y) * kernel_matrix * y).astype(float) # quadratic term 

        # alpha dot y = 0 constraint 
        A = np.transpose(y).astype(float)
        b = 0

        # all alphas between 0 and C constraint
        lb = np.zeros(self.num).astype(float)
        ub = np.full(self.num, self.C).astype(float)

        # Solve (QP): 
        # MIN 1/2 alpha^t P alpha + q^t alpha
        # SUBJECT TO
        # A dot alpha = b
        # lb <= alpha <= ub
        self.alphas = solve_qp(P=P, q=q, A=A, b=b, lb=lb, ub=ub)
 
        # Save support vectors  
        total_bias = 0 #also accumulate what the total bias is 
        for i in range(self.num):
            if (self.alphas[i] > 0): #on a support vector 
                self.nonzero_alphas.append(self.alphas[i])
                self.support_vectors.append(X[i])
                self.support_vector_labels.append(y[i])
                
                #calculate what the prediction would be for this support vector
                #based on other support vectors
                #slightly awkward so that we don't need to do another kernel matrix
                prediction_i = 0  
                for j in range(self.num): 
                    if (self.alphas[j] > 0): #on a support vector 
                        prediction_i += self.alphas[j] * y[j] * kernel_matrix[i][j] 
                total_bias += prediction_i - y[i] 
        self.b = total_bias / len(self.support_vectors) #b is the mean bias for support vecs



    def predict(self, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of examples to classify
            kernel_matrix: an ndarray containing kernel evaluations between the examples and support vectors 
            rows corresp to testing examples, columns corresp to support vectors.

        Returns:
            An array of ints with len = len(X).
        """
        y_hat = np.zeros(len(X)) #initialize the predictions to zero
        for i in range(len(X)): #over examples (rows of kernel matrix)
            for j in range(len(self.support_vectors)): #over support vectors (columns of kernel matrix)
                y_hat[i] +=  self.nonzero_alphas[j] * self.support_vector_labels[j] * kernel_matrix[i][j] #add bias term
        y_hat = np.sign(y_hat).astype(int) 
        return y_hat

        
    def get_support_vectors(self):
        """
        Returns:
            Array of support vectors.
        """
        return self.support_vectors 



#SOME SMALL TESTS 
#TRAINING 
x = ["friend", "love", "bad word 3", "bad word 4"]
y = [1, 1, -1, -1]
kernel_matrix = [[100, 5, 1, 0],
                 [5, 100, 2, 3],
                 [1, 2, 100, 4],
                 [0, 3, 4, 100]]
mod = DualSVM(4, 100)
mod.fit(x, y, kernel_matrix)

#TESTING 
x = ["bad word", "bad word 2"]
y = [-1, -1]
kernel_matrix = [[2, 0, 40, 8],
                 [1, 1, 20, 10]]
preds = mod.predict(x, kernel_matrix)
if((y == preds).all()):
    print("Test 1 passed");
else:
    print("Test 1 failed");


x = ["happy", "cute"]
y = [1, 1]
kernel_matrix = [[100, 57, 1, 2],
                 [500, 350, 20, 10]]
preds = mod.predict(x, kernel_matrix)
if((y == preds).all()):
    print("Test 2 passed");
else:
    print("Test 2 failed");

#TESTING 4
x = ["happy", "cute", "bad", "mean", "nice"]
y = [1, 1, -1, -1, 1]
kernel_matrix = [[100, 57, 1, 2],
                 [1000, 799, 0, 5],
                 [0, 3, 55, 800],
                 [1, 7, 600, 299],
                 [500, 350, 20, 10]]
preds = mod.predict(x, kernel_matrix)
if((y == preds).all()):
    print("Test 3 passed");
else:
    print("Test 3 failed");










