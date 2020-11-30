#trying to do this testcase https://link.springer.com/chapter/10.1007%2F3540530320_2

import qpsolvers as qp
import numpy as np


def function_value(x,q,P):
    return q@x-0.5*x.T@P@x


P = (-100) * np.identity(5).astype(float)
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
#waiting on a response from the qpsolvers github people on why this produces an error - if its obvious to anyone please lmk!
#qpsolvers requires P to be positive definite which in the example case is not ,
#trying somthing with cvxopt.solvers, hope it will work- Yanzong
print(x_star)
print(function_value(x_star,q,P))


