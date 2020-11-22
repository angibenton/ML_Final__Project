import qpsolvers as qp
import numpy as np

#trying to do this testcase https://link.springer.com/chapter/10.1007%2F3540530320_2

P = (-100) * np.identity(5).astype(float)
print("P: " , P.shape)
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

#WHY DOESNT THIS WORK!!!!!!!!!!!!!!! ohmygod

x_star = qp.solve_qp(P = P, q = q, G = G, h = h, lb = lb, ub = ub)

print(x_star)
