# Make a plot of the average energy and specific heat
# based on the Monte Carlo simulation and RBM training

import numpy as np
import matplotlib.pyplot as plt
import pickle


def energy(state):
    en = 0
    N = len(state)
    for i in range(N):
        for j in range(N):
            S = state[i,j]
            nb = state[(i+1)%N, j] + \
                 state[i,(j+1)%N] + \
                 state[(i-1)%N, j] + \
                 state[i,(j-1)%N]
            en += -nb*S
    return en/4.

def magnetization(state):
    return np.sum(state)
    
def compute_observables(temperatures, file_pattern, N, n1, n2):
    nt = len(temperatures)
    E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for i, T in enumerate(temperatures):
        print('-> %i/%i ...'%(i+1, len(temperatures)))
        E1 = M1 = E2 = M2 = 0
        beta =1.0/T
        beta2 = beta*beta
        # 'data/temp_%.4f_size_%i.pickle'
        data = pickle.load(open(file_pattern%(T, N), 'rb'))
        for state in data:
            Ene = energy(state)
            #Mag = magnetization(state)
            E1 = E1 + Ene
            #M1 = M1 + Mag
            #M2 = M2 + Mag*Mag
            E2 = E2 + Ene*Ene
        E[i] = n1*E1
        #M[i] = n1*M1
        C[i] = (n1*E2 - n2*E1*E1)*beta2
        #X[i] = (n1*M2 - n2*M1*M1)*beta
    return E, C

if __name__ == '__main__':

    import config

    ### monte carlo data ###
    mcSteps, N = config.mcSteps, config.N
    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
    mc_temps = np.linspace(config.tmin, config.tmax, config.nt)
    #Emc, Cmc = compute_observables(mc_temps, config.monte_carlo_output,
    #                                N, n1, n2)
    #pickle.dump([Emc, Cmc], open(config.mc_save_plot, 'wb'))
    Emc, Cmc = pickle.load(open(config.mc_save_plot, 'rb'))

    ### RBM with gradient descent ###
    steps, N = config.gd_num_train_points, config.N
    n1, n2  = 1.0/(steps*N*N), 1.0/(steps*steps*N*N) 
    mc_temps = np.linspace(config.tmin, config.tmax, config.nt)
    gd_temps = [mc_temps[i] for i in range(0, len(mc_temps), config.gd_skip)]
    #Egd, Cgd = compute_observables(gd_temps,config.gd_file_format, N, n1, n2)
    #pickle.dump([Egd, Cgd], open(config.gd_save_plot, 'wb'))
    Egd, Cgd = pickle.load(open(config.gd_save_plot, 'rb'))
    
    ### RBM with leapfrog ###
    steps, N = config.lf_num_train_points, config.N
    n1, n2  = 1.0/(steps*N*N), 1.0/(steps*steps*N*N) 
    mc_temps = np.linspace(config.tmin, config.tmax, config.nt)
    lf_temps = [mc_temps[i] for i in range(0, len(mc_temps), config.lf_skip)]
    #Elf, Clf = compute_observables(lf_temps, config.lf_file_format, N, n1, n2)
    #pickle.dump([Elf, Clf], open(config.lf_save_plot, 'wb'))
    Elf, Clf = pickle.load(open(config.lf_save_plot, 'rb'))
    
    ### RBM with rgd ###
    #steps, N = config.rgd_num_train_points, config.N
    #n1, n2  = 1.0/(steps*N*N), 1.0/(steps*steps*N*N) 
    #mc_temps = np.linspace(config.tmin, config.tmax, config.nt)
    #rgd_temps = [mc_temps[i] for i in range(0, len(mc_temps), config.rgd_skip)]
    #Ergd, Crgd = compute_observables(rgd_temps, config.rgd_file_format, N, n1, n2)
    #pickle.dump([Ergd, Crgd], open(config.rgd_save_plot, 'wb'))
    #Ergd, Crgd = pickle.load(open(config.rgd_save_plot, 'rb'))
    
    # RBM with Nesterov
    #steps, N = config.nest_num_train_points, config.N
    #n1, n2  = 1.0/(steps*N*N), 1.0/(steps*steps*N*N) 
    #mc_temps = np.linspace(config.tmin, config.tmax, config.nt)
    #nest_temps = [mc_temps[i] for i in range(0,len(mc_temps),config.nest_skip)]
    #Enest, Cnest = compute_observables(nest_temps, config.nest_file_format,
    #                                N, n1, n2)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axvline(x=2.269, ls='--', color='k', lw=1, dashes=(5, 5))
    ax.plot(mc_temps[1:], Emc[1:], '-d', lw=1.5, color='k', label='Monte Carlo')
    ax.plot(gd_temps[1:], Egd[1:], 's', label='gradient descent', 
                color='orangered')
    ax.plot(lf_temps[1:], Elf[1:], 'o', label='presympletic leapfrog',
                color='royalblue')
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\langle E \rangle / N$")
    ax.legend(loc=4)
    fig.savefig('ising_energy.pdf', bbox_inches='tight')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axvline(x=2.269, ls='--', color='k', lw=1, dashes=(5,5))
    ax.plot(mc_temps[1:], Cmc[1:], '-d', lw=1.5, color='k')
    ax.plot(gd_temps[1:], Cgd[1:], 's', color='orangered')
    ax.plot(lf_temps[1:], Clf[1:], 'o', color='royalblue')
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\langle C \rangle / N$")
    fig.savefig('ising_heat.pdf', bbox_inches='tight')

