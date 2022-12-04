""" 
Optimization solvers with constant damping.
One has to be careful when \gamma is large because the exponentials
become numerically unstable. we corrected this in the leapfrog only
(leapfrog2 below).
We use the gradient norm as stopping criterion.

"""

import numpy as np

def euler1(grad, q0, gamma=1.0, h=0.01, max_iter=1e3, tol=1e-10):
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    norm_gradients = []
    
    for k in range(int(max_iter)):
        
        g = grad(q)
        p = p - h*np.exp(gamma*t)*g
        q = q + h*np.exp(-gamma*t)*p
        t = t + h

        norm_grad = np.linalg.norm(g)
        
        if norm_grad <= tol:
            break
        
        trajectory.append(q)
        norm_gradients.append(norm_grad)
    
    return np.array(norm_gradients), np.array(trajectory)

def leapfrog1(grad, q0, gamma=1.0, h=0.01, max_iter=1e3, tol=1e-10):
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    norm_gradients = []
    g = grad(q)
    
    for k in range(int(max_iter)):
        
        p = p - (h/2)*np.exp(gamma*t)*g
        t = t + h
        q = q + (h/2)*(np.exp(-gamma*(t-h)) + np.exp(-gamma*t))*p
        g = grad(q)
        p = p - (h/2)*np.exp(gamma*t)*g

        norm_grad = np.linalg.norm(g)
        
        if norm_grad <= tol:
            break

        trajectory.append(q)
        norm_gradients.append(norm_grad)
    
    return np.array(norm_gradients), np.array(trajectory)

def leapfrog2(grad, q0, gamma=1.0, h=0.01, max_iter=1e3, tol=1e-10):
    # this is equivalent to the above, however we wrote in terms
    # of the "physical momentum" which allows us to remove undesirable
    # exponentials numerically
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    norm_gradients = []
    g = grad(q)
    mu = np.exp(-gamma*h/2)
    
    for k in range(int(max_iter)):
        
        p = mu*(p - (h/2)*g)
        t = t + h
        q = q + h*np.cosh(gamma*h/2)*p
        g = grad(q)
        p = mu*p - (h/2)*g

        norm_grad = np.linalg.norm(g)
        
        if norm_grad <= tol:
            break

        trajectory.append(q)
        norm_gradients.append(norm_grad)
    
    return np.array(norm_gradients), np.array(trajectory)

def leapfrog3(grad, q0, mu1=0.9, mu2=1, h=0.01, max_iter=1e3, tol=1e-10):
    # this is equivalent to the above, however we wrote in terms
    # of the "physical momentum" which allows us to remove undesirable
    # exponentials numerically
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    norm_gradients = []
    g = grad(q)
    #mu = np.exp(-gamma1*h/2)
    mu = mu1
    cosh = np.cosh(-np.log(mu1))
    
    for k in range(int(max_iter)):
        
        p = mu*(p - (h/2)*(mu2)*g)
        t = t + h
        q = q + h*cosh*p
        g = grad(q)
        p = mu*p - (h/2)*(mu2)*g

        norm_grad = np.linalg.norm(g)
        
        if norm_grad <= tol:
            break

        trajectory.append(q)
        norm_gradients.append(norm_grad)
    
    return np.array(norm_gradients), np.array(trajectory)

def yoshida1(grad, q0, gamma=1, h=0.01, max_iter=1e3, tol=1e-10):
    t = 0
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    norm_gradients = []
    g = grad(q)
    kappa = np.power(2, 1/3)
    tau0 = 1/(2-kappa)
    tau1 = -kappa/(2-kappa)
    
    for k in range(int(max_iter)):

        s = h*tau0
        p = p - (s/2)*np.exp(gamma*t)*g
        t = t + s
        q = q + (s/2)*(np.exp(-gamma*(t-s)) + np.exp(-gamma*t))*p
        g = grad(q)
        p = p - (s/2)*np.exp(gamma*t)*g

        s = h*tau1
        p = p - (s/2)*np.exp(gamma*t)*g
        t = t + s
        q = q + (s/2)*(np.exp(-gamma*(t-s)) + np.exp(-gamma*t))*p
        g = grad(q)
        p = p - (s/2)*np.exp(gamma*t)*g
        
        s = h*tau0
        p = p - (s/2)*np.exp(gamma*t)*g
        t = t + s
        q = q + (s/2)*(np.exp(-gamma*(t-s)) + np.exp(-gamma*t))*p
        g = grad(q)
        p = p - (s/2)*np.exp(gamma*t)*g
        
        norm_grad = np.linalg.norm(g)
        
        if norm_grad <= tol:
            break

        trajectory.append(q)
        norm_gradients.append(norm_grad)
    
    return np.array(norm_gradients), np.array(trajectory)

def nesterov(grad, q0, gamma=1.0, h=0.01, max_iter=1e3, tol=1e-10):
    q = q0
    p = np.zeros(q.shape)
    trajectory = []
    norm_gradients = []

    mu = np.exp(-gamma*h)
    
    for k in range(int(max_iter)):
        
        #mu = k/(k+3) # for convex problems
        oldq = q
        q = q + h*mu*p
        g = grad(q)
        p = mu*p - h*g
        q = oldq + h*p
        
        norm_grad = np.linalg.norm(g)
        
        if norm_grad <= tol:
            break

        trajectory.append(q)
        norm_gradients.append(norm_grad)
    
    return np.array(norm_gradients), np.array(trajectory)

