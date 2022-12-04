"""Implementing the presymplectic discretizations.
It assumes a gradient is passed as an input.


below we have the following variables:

    grad:       function of x that computes the gradient
    q0:         initial position
    eta1:       function of time t that computes the damping term
    eta2:       function of time t that computes another damping term
    h:          step size
    Minv:       the inverse of a "mass matrix"
    max_iter:   maximum number of iterations
    tol:        tolerance for the norm of the gradient (stopping criteria)

"""

import numpy as np


def euler1(grad, q0, eta1, eta2, h=0.01, Minv=None, max_iter=1e3, tol=1e-10):
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    
    if Minv is None:
        Minv = np.eye(q.shape[0])

    for k in range(int(max_iter)):
        
        g = grad(q)
        p = p - h*np.exp(eta2(t))*g
        q = q + h*np.exp(-eta1(t))*Minv.dot(p)
        t = t + h
        
        if np.linalg.norm(g) <= tol:
            break
        
        trajectory.append(q)
    
    return np.array(trajectory)

def leapfrog1(grad, q0, eta1, eta2, h=0.01, Minv=None, 
                max_iter=1e3, tol=1e-10):
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    g = grad(q)
    
    if Minv is None:
        Minv = np.eye(q.shape[0])
    
    for k in range(int(max_iter)):
        
        p = p - (h/2)*np.exp(eta2(t))*g
        t = t + h
        q = q + (h/2)*(np.exp(-eta1(t-h)) + np.exp(-eta1(t)))*Minv.dot(p)
        g = grad(q)
        p = p - (h/2)*np.exp(eta2(t))*g

        if np.linalg.norm(g) <= tol:
            break

        trajectory.append(q)
    
    return np.array(trajectory)

def leapfrog2(grad, q0, eta1, eta2, h=0.01, Minv=None, 
                max_iter=1e3, tol=1e-10):
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    
    if Minv is None:
        Minv = np.eye(q.shape[0])
    
    for k in range(int(max_iter)):
        
        t = t + h/2
        q = q + (h/2)*np.exp(-eta1(t))*Minv.dot(p)
        g = grad(q)
        p = p - h*np.exp(eta2(t))*g
        q = q + (h/2)*np.exp(-eta1(t))*Minv.dot(p)
        t = t + h/2

        if np.linalg.norm(g) <= tol:
            break

        trajectory.append(q)
    
    return np.array(trajectory)

def yoshida1(grad, q0, eta1, eta2, h=0.01, Minv=None, max_iter=1e3, tol=1e-10):
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    g = grad(q)
    kappa = np.power(2, 1/3)
    tau0 = 1/(2-kappa)
    tau1 = -kappa/(2-kappa)
    
    if Minv is None:
        Minv = np.eye(q.shape[0])
    
    for k in range(int(max_iter)):

        s = h*tau0
        p = p - (s/2)*np.exp(eta2(t))*g
        t = t + s
        q = q + (s/2)*(np.exp(-eta1(t-s)) + np.exp(-eta1(t)))*Minv.dot(p)
        g = grad(q)
        p = p - (s/2)*np.exp(eta2(t))*g

        s = h*tau1
        p = p - (s/2)*np.exp(eta2(t))*g
        t = t + s
        q = q + (s/2)*(np.exp(-eta1(t-s)) + np.exp(-eta1(t)))*Minv.dot(p)
        g = grad(q)
        p = p - (s/2)*np.exp(eta2(t))*g
        
        s = h*tau0
        p = p - (s/2)*np.exp(eta2(t))*g
        t = t + s
        q = q + (s/2)*(np.exp(-eta1(t-s)) + np.exp(-eta1(t)))*Minv.dot(p)
        g = grad(q)
        p = p - (s/2)*np.exp(eta2(t))*g

        if np.linalg.norm(g) <= tol:
            break

        trajectory.append(q)
    
    return np.array(trajectory)

def yoshida2(grad, q0, eta1, eta2, h=0.01, Minv=None, max_iter=1e3, tol=1e-10):
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    g = grad(q)
    kappa = np.power(2, 1/3)
    tau0 = 1/(2-kappa)
    tau1 = -kappa/(2-kappa)
    
    if Minv is None:
        Minv = np.eye(q.shape[0])
    
    for k in range(int(max_iter)):

        s = h*tau0
        t = t + s/2
        q = q + (s/2)*np.exp(-eta1(t))*Minv.dot(p)
        g = grad(q)
        p = p - s*np.exp(eta2(t))*g
        q = q + (s/2)*np.exp(-eta1(t))*Minv.dot(p)
        t = t + s/2

        s = h*tau1
        t = t + s/2
        q = q + (s/2)*np.exp(-eta1(t))*Minv.dot(p)
        g = grad(q)
        p = p - s*np.exp(eta2(t))*g
        q = q + (s/2)*np.exp(-eta1(t))*Minv.dot(p)
        t = t + s/2
        
        s = h*tau0
        t = t + s/2
        q = q + (s/2)*np.exp(-eta1(t))*Minv.dot(p)
        g = grad(q)
        p = p - s*np.exp(eta2(t))*g
        q = q + (s/2)*np.exp(-eta1(t))*Minv.dot(p)
        t = t + s/2

        if np.linalg.norm(g) <= tol:
            break

        trajectory.append(q)
    
    return np.array(trajectory)

def nesterov(grad, q0, mu=0.9, h=0.01, max_iter=1e3, tol=1e-10):
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    
    for k in range(int(max_iter)):
        
        #mu = k/(k+3) # for convex problems
        #otherwise mu = e^{-gamma h}
        oldq = q
        q = q + h*mu*p
        g = grad(q)
        p = mu*p - h*g
        q = oldq + h*p
        
        trajectory.append(q)

        if np.linalg.norm(g) <= tol:
            break
    
    return np.array(trajectory)


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    # correlated quadratic function
    d = 10          # dimension
    rho = 0.96      # correlation coefficient
    Q = np.array([np.power(rho, np.abs(i-j))
                  for i in range(1, d+1, 1) for j in range(1, d+1, 1)])
    Q = Q.reshape((d,d))
    Qinv = np.linalg.inv(Q)
    func = lambda x: np.dot(x, np.dot(Qinv, x)) # quadratic objective
    grad = lambda x: Q.dot(x)   # gradient
    q0 = np.ones(d) # initial position
    
    mi=1e5          # maximum iteration
    tol=1e-10       # tolerance for convergence
    gamma = 0.4       # damping coefficient
    
    # damping functions
    eta1 = lambda t: gamma*t 
    eta2 = lambda t: gamma*t
    
    results = {}
    
    iterates = euler1(grad, q0, eta1, eta2, h=0.6, max_iter=mi, tol=tol)
    ys = [func(a) for a in iterates]
    results['Euler'] = ys
    
    iterates = leapfrog2(grad, q0, eta1, eta2, h=0.6, max_iter=mi, tol=tol)
    ys = [func(a) for a in iterates]
    results['leapfrog'] = ys
    
    iterates = yoshida2(grad, q0, eta1, eta2, h=0.5, max_iter=mi, tol=tol)
    ys = [func(a) for a in iterates]
    results['Suzuki-Yoshida'] = ys
    
    h_nest = 0.3
    mu = np.exp(-gamma*h_nest)
    iterates = nesterov(grad, q0, mu, h_nest, max_iter=mi, tol=tol)
    ys = [func(a) for a in iterates]
    results['Nesterov'] = ys

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    
    for algo, ys in results.items():
        ax.plot(ys, label=algo)
    ax.legend(loc=0)
    
    fig.savefig('test.pdf')

