""" 
Solve a random quadratic problem
We show a convergence rate plot and a phase diagram.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
import pickle

import optimizers2 as solver


def random_quadratic_many():
    d = 1000
    #M = ortho_group.rvs(dim=d)
    #A = M.T.dot(np.diag(np.random.uniform(0.1, 10, d)).dot(M))
    #g = np.random.normal(0, 10, d)
    p = int(d/0.8)

    lf = []
    nest = []
    lf_new = []
    for e in range(50):
        M = np.random.normal(0, 1, (p,d))
        Q = M.T.dot(M)/d
        g = np.zeros(d)
        func = lambda x: 0.5*x.dot(Q.dot(x)) + g.dot(x)
        grad = lambda x: Q.dot(x) + g
        x0 = 1*np.ones(d)
        gamma = 0.7
        grads1, _ =  solver.leapfrog2(grad,x0,h=0.9,gamma=gamma,
                        max_iter=1000,tol=1e-15)
        grads2, _ =  solver.nesterov(grad,x0,h=0.5,gamma=gamma,
                        max_iter=1000,tol=1e-15)
        #grads3, _ =  solver.leapfrog3(grad,x0,h=0.5,gamma1=gamma1, 
        #                gamma2=gamma2, max_iter=1000,tol=1e-15)
        lf.append(grads1)
        nest.append(grads2)
        #lf_new.append(grads3)
    lf = np.array(lf)
    nest = np.array(nest)
    #lf_new = np.array(lf_new)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ys1 = lf.mean(axis=0)
    xs1 = range(len(ys1))
    std1 = lf.std(axis=0)
    ax.plot(xs1, ys1, label=r'leapfrog', marker='s', color='royalblue',
            markevery=0.1)
    ax.fill_between(xs1, ys1-std1, ys1+std1, alpha=.3, color='royalblue')
    ys2 = nest.mean(axis=0)
    xs2 = range(len(ys2))
    std2 = nest.std(axis=0)
    ax.plot(ys2, label=r'Nesterov', marker='D', color='orangered',
            markevery=0.1)
    ax.fill_between(xs2, ys2-std2, ys2+std2, alpha=.3, color='orangered')
    #ys3 = lf_new.mean(axis=0)
    #xs3 = range(len(ys3))
    #std3 = lf_new.std(axis=0)
    #ax.plot(ys3, label=r'leapfrog (rescaled)', marker='h', markevery=0.1)
    #ax.fill_between(xs3, ys3-std3, ys3+std3, alpha=.3)
    ax.set_ylabel(r'$\| \nabla f(q_\ell)\|$')
    ax.set_xlabel(r'$\ell$')
    ax.legend(loc=0)
    fig.savefig('randomquadratic2.pdf', bbox_inches='tight')

def random_quadratic_phase():
    d = 1000
    p = int(d/0.8)
    hmin = 0.01; hmax = 1.1;
    gmin = 0.0; gmax = 5;
    M = np.random.normal(0, 1, (p,d))
    Q = M.T.dot(M)/d
    g = np.zeros(d)
    func = lambda x: 0.5*x.dot(Q.dot(x)) + g.dot(x)
    grad = lambda x: Q.dot(x) + g
    x0 = 1*np.ones(d)

    numh = 50
    numg = 50
    hrange = np.linspace(hmin, hmax, numh)
    gammarange = np.linspace(gmax, gmin, numg)
  
    """
    X = np.ones((numg,numh))
    Y = np.ones((numg,numh))
    for i, gamma in enumerate(gammarange):
        for j, h in enumerate(hrange):
            grads1, _ =  solver.leapfrog2(grad,x0,h=h,gamma=gamma,
                                max_iter=800,tol=1e-3)
            grads2, _ =  solver.nesterov(grad,x0,h=h,gamma=gamma+100,
                                max_iter=800,tol=1e-3)
            X[i,j] = grads1[-1]
            Y[i,j] = grads2[-1]
    X = np.nan_to_num(X, nan=1e100, posinf=1e100, neginf=1e100)
    Y = np.nan_to_num(Y, nan=1e100, posinf=1e100, neginf=1e100)
    X[X>1] = 1
    Y[Y>1] = 1
    print(X)
    print(Y)

    pickle.dump(X, open('X.pickle', 'wb'))
    pickle.dump(Y, open('Y.pickle', 'wb'))
    """

    X = pickle.load(open('X.pickle', 'rb'))
    Y = pickle.load(open('Y.pickle', 'rb'))

    fig, (ax1, ax2)= plt.subplots(1, 2, figsize=(12, 5))
                        #sharey=True, sharex=True)
    #cm = plt.cm.magma_r
    cm = plt.cm.CMRmap_r
    #cm = plt.cm.afmhot_r
    #cm = plt.cm.hot_r
    #cm = plt.cm.pink_r
    #cm = plt.cm.bone_r
    #cm = plt.cm.cubehelix_r
    
    a = ax1.matshow(X, cmap=cm)

    xticks = np.linspace(hmin, hmax, 6)
    xtickslabel = ['']+['%.2f'%x for x in xticks]
    ax1.set_xticklabels(xtickslabel)
    
    yticks = np.linspace(gmax, gmin, 6)
    ytickslabel = ['']+['%.2f'%x for x in yticks]
    ax1.set_yticklabels(ytickslabel)
    ax1.tick_params(axis="x",bottom=True,top=False,
                            labelbottom=True,labeltop=False)
    ax1.set_xlabel(r'$h$')
    ax1.set_ylabel(r'$\gamma$')
    ax1.set_title('leapfrog', pad=-3)
    
    b = ax2.matshow(Y, cmap=cm)
    ax2.set_xticklabels(xtickslabel)
    ax2.set_yticklabels(ytickslabel)
    ax2.tick_params(axis="x",bottom=True,top=False,
                            labelbottom=True,labeltop=False)
    #ax.set_xticklabels(['']+['%.2f'%x for x in np.linspace(0.01, 1.1, 6)])
    #ax.set_yticklabels(['']+['%.2f'%x for x in np.linspace(2, 0.1, 6)])
    #ax.set_yticklabels([''])
    #ax.tick_params(axis="x",bottom=True,top=False,
    #                        labelbottom=True,labeltop=False)
    ax2.set_xlabel(r'$h$')
    #ax2.set_ylabel(r'$\gamma$')
    ax2.set_yticklabels([])
    ax2.set_title('Nesterov', pad=-3)

    plt.subplots_adjust(wspace=-0.20, hspace=0)

    cbar_ax = fig.add_axes([0.88, 0.11, 0.03, 0.77])
    fig.colorbar(a, cax=cbar_ax)
    
    fig.savefig('randomphase.pdf', bbox_inches='tight')


if __name__ == '__main__':
    #random_quadratic_many()
    random_quadratic_phase()
