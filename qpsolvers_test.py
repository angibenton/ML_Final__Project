#trying to do this testcase https://link.springer.com/chapter/10.1007%2F3540530320_2

import qpsolvers as qp
import numpy as np

<<<<<<< HEAD
P = (100) * np.identity(5).astype(float)
=======

def function_value(x,q,P):
    return q@x-0.5*x.T@P@x


P = (-100) * np.identity(5).astype(float)
>>>>>>> 84abf2ae1027b20c42ae420b2b079885fa56dc4a
print("P: " , P.shape)
print(P.dtype)
q = np.array([42, 44, 45, 47, 47.5]).astype(float)
print("q: " , q.shape)
G = np.array([[20, 12, 11, 7, 4]]).astype(float)
print("G: " , G.shape)
h = np.array([40]).astype(float)
print("h: " , h.shape)
lb = np.zeros((5,)).astype(float)
print("lb: " , lb.shape)
ub = np.ones((5,)).astype(float)
print("ub: " , ub.shape)

x_star = qp.solve_qp(P = P, q = q, G = G, h = h, lb = lb, ub = ub)
<<<<<<< HEAD
=======
#waiting on a response from the qpsolvers github people on why this produces an error - if its obvious to anyone please lmk!
#qpsolvers requires P to be positive definite which in the example case is not ,
#if we can keep our matrix to be positive definite this will not be a problem
>>>>>>> 84abf2ae1027b20c42ae420b2b079885fa56dc4a
print(x_star)
print(function_value(x_star,q,P))


