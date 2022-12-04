# test suite using several optimization functions
# See https://arxiv.org/pdf/1308.4008.pdf for details

import numpy as np
import matplotlib.pyplot as plt

from opt import OPT
import optimizers2 as solver


def test(x0, func, grad, par_search, 
         max_iter_tune=50, max_evals=500,
         max_iter=100, tol=1e-10, fname='plot.pdf'):
    
    def par_range(*names):
        d = par_search # dictionary with par name and range
        return {n: d[n] for n in names}

    #euler = OPT(grad, x0)
    #euler.tol = tol
    #euler.set_solver(solver.euler1)
    #euler.max_iter = max_iter_tune # max iterations to tune
    #euler.tune(par_range('h','gamma'), max_evals=max_evals)
    #euler.max_iter = max_iter # max iterations to solve with best params
    #euler.solve()
    #print('** Euler **')
    #print('params:', euler.best)
    #print('solution:', func(euler.states[-1]))
    #print()
    
    lf = OPT(grad, x0)
    lf.tol = tol
    lf.set_solver(solver.leapfrog2)
    lf.max_iter = max_iter_tune # max iterations to tune
    lf.tune(par_range('h', 'gamma'), max_evals=max_evals)
    lf.max_iter = max_iter # max iterations to solve with best params
    lf.solve()
    print()
    print('* Leapfrog ******************************************************')
    print('params:', lf.best)
    print('solution:', func(lf.states[-1]))
    print()
    
    lfr = OPT(grad, x0)
    lfr.tol = tol
    lfr.set_solver(solver.leapfrog3)
    lfr.max_iter = max_iter_tune # max iterations to tune
    lfr.tune(par_range('h', 'mu1', 'mu2'), max_evals=max_evals)
    lfr.max_iter = max_iter # max iterations to solve with best params
    lfr.solve()
    print()
    print('* Leapfrog rescaled *********************************************')
    print('params:', lfr.best)
    print('solution:', func(lfr.states[-1]))
    print()
    
    #yo = OPT(grad, x0)
    #yo.tol = tol
    #yo.set_solver(solver.yoshida1)
    #yo.max_iter = max_iter_tune # max iterations to tune
    #yo.tune(par_range('h','gamma'), max_evals=max_evals)
    #yo.max_iter = max_iter # max iterations to solve with best params
    #yo.solve()
    #print('** Yoshida **')
    #print('params:', yo.best)
    #print('solution:', func(yo.states[-1]))
    #print()
    
    nest = OPT(grad, x0)
    nest.tol = tol
    nest.set_solver(solver.nesterov)
    nest.max_iter = max_iter_tune # max iterations to tune
    nest.tune(par_range('h', 'gamma'), max_evals=max_evals)
    nest.max_iter = max_iter # max iterations to solve with best params
    nest.solve()
    print()
    print('* Nesterov ******************************************************')
    print('params:', nest.best)
    print('solution:', func(nest.states[-1]))
    print()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    scores1 = [func(x) for x in lf.states]
    ax.plot(scores1, label='leapfrog', marker='o', markevery=.1)
    scores3 = [func(x) for x in lfr.states]
    ax.plot(scores3, label='leapfrog (rescaled)', marker='h', markevery=.1)
    scores2 = [func(x) for x in nest.states]
    ax.plot(scores2, label='Nesterov', marker='s', markevery=.1)
    ax.set_ylabel(r'$\nabla f(q_\ell)$')
    ax.set_xlabel(r'$\ell$')
    ax.legend(loc=0)
    fig.savefig(fname, bbox_inches='tight')


##############################################################################
# several test functions below

def ros(x):
    xp = x[1:]
    xm = x[:-1]
    return (100*(xp-xm**2)**2 + (1-xm)**2).sum()

def ros_grad(x):
    G = np.zeros(len(x))
    G[1:-1] = -400*(x[2:]-x[1:-1]**2)*x[1:-1] - \
                    2*(1-x[1:-1]) + 200*(x[1:-1]-x[:-2]**2)
    G[0] = -400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0])
    G[-1] = 200*(x[-1]-x[-2]**2)
    return G

def beale(x):
    a = x[0]
    b = x[1]
    return (1.5-a+a*b)**2 + (2.25-a+a*(b**2))**2+(2.625-a+a*(b**3))**2

def beale_grad(x):
    a = x[0]
    b = x[1]
    xcomp = 2*(1.5-a+a*b)*(-1+b) + \
            2*(2.25-a+a*(b**2))*(-1+(b**2)) + \
            2*(2.625-a+a*(b**3))*(-1+(b**3))
    ycomp = 2*(1.5-a+a*b)*(a) + \
            2*(2.25-a+a*(b**2))*(2*a*b) + \
            2*(2.625-a+a*(b**3))*(3*a*(b**2))
    return np.array([xcomp, ycomp])

def rastrigin(x):
    A = 10
    return A*len(x) + np.sum(x**2-A*np.cos(2*np.pi*x))

def rastrigin_grad(x):
    A = 10
    return 2*x + A*2*np.pi*np.sin(2*np.pi*x)

def ackley(x):
    return -200*np.exp(-0.02*np.linalg.norm(x)) + 200

def ackley_grad(x):
    return -200*np.exp(-0.02*np.linalg.norm(x))*(-0.02*x/np.linalg.norm(x))

def ackley3(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    n = len(x)
    return -a*np.exp(-b*np.linalg.norm(x)/np.sqrt(n)) - \
            np.exp(np.sum(np.cos(c*x))/n) + a + np.exp(1)

def ackley3_grad(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    n = len(x)
    return a*b*np.exp(-b*np.linalg.norm(x)/np.sqrt(n))*x/(np.sqrt(n)*np.linalg.norm(x)) + np.exp(np.sum(np.cos(c*x)))*c*np.sin(c*x)/n
            
def goldstein(z):
    x = z[0]
    y = z[1]
    return (1 + (1 + x + y)**2*(19 - 14*x + 3*x**2 - 14*y + 6*x*y +  \
      3*y**2))*(30 + (2*x - 3*y)**2*(18 - 32*x + 12*x**2 + 48*y - \
      36*x*y + 27*y**2))

def goldstein_grad(z):
    x = z[0]
    y = z[1]
    gx = 24*(-1 + 2*x - 3*y)*(2*x - 3*y)*(2*x - \
    3*(1 + y))*(1 + (1 + x + y)**2*(19 + 3*x**2 + y*(-14 + 3*y) + \
       2*x*(-7 + 3*y))) + 12*(-2 + x + y)*(-1 + x + y)*(1 + x + \
        y)*(30 + (2*x - 3*y)**2*(18 + 12*x**2 - 4*x*(8 + 9*y) + \
        3*y*(16 + 9*y)))
    gy = -36*(-1 + 2*x - 3*y)*(2*x - 3*y)*(2*x - \
    3*(1 + y))*(1 + (1 + x + y)**2*(19 + 3*x**2 + y*(-14 + 3*y) + \
       2*x*(-7 + 3*y))) + \
 12*(-2 + x + y)*(-1 + x + y)*(1 + x + \
    y)*(30 + (2*x - 3*y)**2*(18 + 12*x**2 - 4*x*(8 + 9*y) + \
       3*y*(16 + 9*y)))
    return np.array([gx,gy])

def booth(z):
    x = z[0]
    y = z[1]
    return (x+2*y-7)**2 + (2*x+y-5)**2 

def booth_grad(z):
    x = z[0]
    y = z[1]
    return np.array([2*(x+2*y-7)*(1) + 4*(2*x+y-5), 
                     2*(x+2*y-7)*(2) + 2*(2*x+y-5)])

def bukin(z):
    x, y = z
    return 100*np.sqrt(np.abs(y-0.01*x**2)) + 0.01*np.abs(x+10)

def bukin_grad(z):
    x, y = z
    gx = 50*np.sign(y-0.01*x**2)*(-0.02*x)/np.sqrt(np.abs(y-0.01*x**2)) + \
            0.01*np.sign(x+10)
    gy = 50*np.sign(y-0.01*x**2)/np.sqrt(np.abs(y-0.01*x**2))
    return np.array([gx,gy])

def matyas(z):
    x, y = z
    return 0.26*(x**2+y**2)-0.48*x*y

def matyas_grad(z):
    x, y = z
    return np.array([0.26*2*x-0.48*y, 0.26*2*y-0.48*x]) 

def levi(z):
    x, y = z
    return (np.sin(3*np.pi*x))**2 + (x-1)**2*(1+(np.sin(3*np.pi*y))**2) \
            + (y-1)**2*(1+(np.sin(2*np.pi*y))**2)

def levi_grad(z):
    x, y = z
    return np.array([
            2*np.sin(3*np.pi*x)*np.cos(3*np.pi*x)*3*np.pi + \
            2*(x-1)*(1+(np.sin(3*np.pi*y))**2), 
            (x-1)**2*(2*np.sin(3*np.pi*y)*np.cos(3*np.pi*y)*3*np.pi) + \
            2*(y-1)*(1+(np.sin(2*np.pi*y))**2) + \
            (y-1)**2*(2*np.sin(2*np.pi*y)*np.cos(2*np.pi*y)*2*np.pi)])

def camel(z):
    x, y = z
    return 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

def camel_grad(z):
    x, y = z
    g1 = 4*x - 1.05*4*x**3 + x**5 + y
    g2 = x + 2*y
    return np.array([g1, g2])

def styblinski(x):
    return np.sum(x**4 - 16*x**2 + 5*x)/2 

def styblinski_grad(x):
    return (4*x**3 - 32*x + 5.)/2.

def brent(x):
    return (x[0]+10)**2 + (x[1]+10)**2 + np.exp(-np.linalg.norm(x)**2)

def brent_grad(x):
    return 2*(x+10) + np.exp(-np.linalg.norm(x)**2)*(-2*x)

def chung_reynolds(x):
    return np.linalg.norm(x)**4

def chung_reynolds_grad(x):
    return 4*np.linalg.norm(x)**2*(x)

def quartic(x):
    i = np.array(range(1, len(x)+1, 1))
    return np.sum(i*x**4)

def quartic_grad(x):
    i = np.array(range(1, len(x)+1, 1))
    return i*4*x**3

def schwefel23(x):
    return np.sum(x**10)

def schwefel23_grad(x):
    return 10*x**9

def qing(x):
    i = np.array(range(1, len(x)+1, 1))
    return np.sum((x**2 - i)**2)

def qing_grad(x):
    i = np.array(range(1, len(x)+1, 1))
    return 4*(x**2-i)*x

def sumsquares(x):
    i = np.array(range(1, len(x)+1, 1))
    return np.sum(i*(x**2))

def sumsquares_grad(x):
    i = np.array(range(1, len(x)+1, 1))
    return 2*i*x

def zakharov(x):
    i = np.array(range(1, len(x)+1, 1))
    a = np.sum(x**2)
    b = (.5*np.sum(i*x))**2
    c = (.5*np.sum(i*x))**4
    return a+b+c

def zakharov_grad(x):
    i = np.array(range(1, len(x)+1, 1))
    a = 2*x
    b = np.sum(i*x)*0.5*i
    c = 4*((0.5*np.sum(i*x))**3)*0.5*i
    return a+b+c

def trid(x):
    d = len(x)
    fstar = -d*(d+4)*(d-1)/6.
    return np.sum((x-1)**2) - np.sum(x[1:]*x[:-1]) - fstar

def trid_grad(x):
    xx = np.zeros(len(x)+2)
    xx[1:-1] = x
    return 2*(x-1) - xx[2:] - xx[:-2]

def rosenbrock(x):
    xp = x[1:]
    xm = x[:-1]
    return (100*(xp-xm**2)**2 + (1-xm)**2).sum()

def rosenbrock_grad(x):
    G = np.zeros(len(x))
    G[1:-1] = -400*(x[2:]-x[1:-1]**2)*x[1:-1] - \
                    2*(1-x[1:-1]) + 200*(x[1:-1]-x[:-2]**2)
    G[0] = -400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0])
    G[-1] = 200*(x[-1]-x[-2]**2)
    return G




##############################################################################
if __name__ == '__main__':

    from scipy.stats import ortho_group

    """
    # random quadratic
    d = 2000
    #M = ortho_group.rvs(dim=d)
    #A = M.T.dot(np.diag(np.random.uniform(0, 10, d)).dot(M))
    #g = np.random.normal(0, 1, d)
    p = int(d/2)
    M = np.random.normal(0, 1, (p, d))
    Q = M.T.dot(M)/d
    g = np.zeros(d)
    quadfunc = lambda x: 0.5*x.dot(Q.dot(x)) + g.dot(x)
    quad_grad = lambda x: Q.dot(x) + g
    x0 = 1*np.ones(d)
    pars = {'h': [1e-4, 9e-1], 'gamma': [0.1, 5]}
    test(x0, quadfunc, quad_grad, pars,
         max_iter_tune=50, max_evals=200,
         max_iter=300, tol=1e-10, fname='randomsquare.pdf')
    """
    
    """
    # correlated quadratic
    rho = 0.9
    d = 1000
    Q = [np.power(rho, np.abs(i-j))
            for i in range(1,d+1) for j in range(1,d+1)]
    Q = np.array(Q)
    Q = Q.reshape((d,d))
    Q = np.linalg.inv(Q)
    corrfunc = lambda x: 0.5*(x.dot(Q.dot(x)))
    corr_grad = lambda x: Q.dot(x)
    x0 = 5*np.ones(d)
    pars = {'h':[1e-4, 1], 'gamma':[0.1, 5]}
    test(x0, corrfunc, corr_grad, pars,
         max_iter_tune=400, max_evals=500,
         max_iter=1000, tol=1e-10, fname='corrsquare.pdf')
    """

    """
    x0 = 1*np.ones(1000)
    pars = {'h': [1e-3, 5e-1], 'gamma': [0.1, 4]}
    test(x0, sumsquares, sumsquares_grad, pars,
         max_iter_tune=300, max_evals=300,
         max_iter=800, tol=1e-10, fname='sumsquares.pdf')
    """

    #x0 = np.random.uniform(-2,2,100)
    x0 = -1*np.ones(100)
    pars = {'h': [1e-4, 1], 'gamma': [0.01, 5], 
            'mu1': [0.5, 1], 'mu2':[0.1,5]}
    test(x0, ros, ros_grad, pars,
         max_iter_tune=2000, max_evals=200,
         max_iter=3000, tol=1e-15, fname='test.pdf')

