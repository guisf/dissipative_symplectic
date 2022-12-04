import numpy as np
import pickle
import matplotlib.pyplot as plt

import config
import rbm_grad
import rbm_leap
import rbm_nest
import rbm_rgd

if __name__ == "__main__":

    temps = np.linspace(config.tmin, config.tmax, config.nt)
    print(temps)
    
    T = 2.666666
    N = config.N
    mi = 2000

    data = pickle.load(open(config.monte_carlo_output%(T, N), 'rb'))
    Z = np.array([d.reshape(N*N) for d in data])
    Z[Z==-1] = 0
    idx = np.random.choice(range(Z.shape[0]), config.lf_num_train_points,
                           replace=False)
    X = Z[idx]
    
    # training RBM
    rbm = rbm_grad.RBM(n_hidden=config.gd_n_hidden,
              batch_size=config.gd_batch_size,
              learning_rate=5e-3,
              n_iterations=mi)
    rbm.fit(X)
    gd_err = rbm.training_errors

    # training RBM
    rbm = rbm_leap.RBM(n_hidden=config.lf_n_hidden,
              batch_size=config.lf_batch_size,
              learning_rate=2.5e-2,
              mu=0.98,
              n_iterations=mi)
    rbm.fit(X)
    lf_err = rbm.training_errors
    
    #rbm = rbm_nest.RBM(n_hidden=config.nest_n_hidden,
    #          batch_size=config.nest_batch_size,
    #          learning_rate=2.8e-2,
    #          mu=0.98,
    #          n_iterations=2000)
    #rbm.fit(X)
    #nest_err = rbm.training_errors
    
    rbm = rbm_rgd.RBM(n_hidden=config.lf_n_hidden,
              batch_size=config.lf_batch_size,
              learning_rate=2.8e-2,
              mu=0.99,
              c=5,
              n_iterations=mi)
    rbm.fit(X)
    rgd_err = rbm.training_errors
    
    #gd_err = pickle.load(open(config.gd_conv_format%(T, N), 'rb'))
    #lf_err = pickle.load(open(config.lf_conv_format%(T, N), 'rb'))
    #nest_err = pickle.load(open(config.nest_conv_format%(T, config.N), 'rb'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(gd_err, color='b', label='gradient descent')
    ax.plot(lf_err, color='r', label='classical leapfrog')
    #ax.plot(nest_err, label='Nesterov')
    ax.plot(rgd_err, color='c', label='relativistic leapfrog')
    ax.set_ylabel('training loss')
    ax.set_xlabel('iteration')
    ax.legend()
    #ax.plot(nest_err)

    fig.savefig('ising_singleT2.pdf', bbox_inches='tight')
