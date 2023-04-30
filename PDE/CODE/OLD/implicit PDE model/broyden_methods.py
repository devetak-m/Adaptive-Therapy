import numpy as np

def broyden_method_sparse(F, x0, zeros, maximum_iterations = 100, tolerance = 1e-6, verbose = False):
    return None

def broyden_method_good(F, x0, J_inverse = None, maximum_iterations = 100, tolerance = 1e-6, verbose = False):  
    # initialize
    y0 = F(x0)
    # compute inverse of J
    if J_inverse is None:
        J_inverse = np.eye(len(x0))
    # compute the error
    error = np.linalg.norm(y0)

    # main loop
    for i in range(maximum_iterations):
        d = -J_inverse.dot(y0)
        x = x0 + d
        y = F(x)
        dF = y - y0
        u = J_inverse.dot(dF)
        J_inverse = J_inverse + np.dot(((d-u).dot(d.T)), J_inverse) / np.dot(d.T,u)
        # u = (d + np.dot(J_inverse,dF))/(np.dot(dF.T,dF))
        # J_inverse = J_inverse + np.dot(u,dF.T)
        x0 = x
        y0 = y
        error = np.linalg.norm(y)
        if error < tolerance:
            return x, error, i
    
    return x, error, i