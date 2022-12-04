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
    
    T = 2.83333
    N = config.N
    mi = 2000

    gd_err = pickle.load(open(config.gd_conv_format%(T, N), 'rb'))
    lf_err = pickle.load(open(config.lf_conv_format%(T, N), 'rb'))
    rgd_err = pickle.load(open(config.rgd_conv_format%(T, N), 'rb'))
    #nest_err = pickle.load(open(config.nest_conv_format%(T, config.N), 'rb'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(gd_err, label='gradient descent', color='orangered')
    ax.plot(lf_err, label='classical leapfrog', color='royalblue')
    #ax.plot(nest_err, label='Nesterov')
    ax.plot(rgd_err, label='relativistic leapfrog', color='mediumseagreen')
    ax.set_ylabel('training loss')
    ax.set_xlabel('iteration')
    ax.legend()
    #ax.plot(nest_err)

    fig.savefig('ising_convergence.pdf', bbox_inches='tight')
