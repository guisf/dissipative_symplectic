"""
Generate data using Monte Carlo with Metropolis update

"""

import numpy as np
import pickle


def initial_state(N):
    return 2*np.random.randint(2, size=(N,N))-1

def mcmove(state, beta):
    N = len(state)
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s =  state[a, b]
            nb = state[(a+1)%N,b] + \
                 state[a,(b+1)%N] + \
                 state[(a-1)%N,b] + \
                 state[a,(b-1)%N]
            cost = 2*s*nb
            if cost < 0:
                s *= -1
            elif np.random.rand() < np.exp(-cost*beta):
                s *= -1
            state[a, b] = s
    return state


if __name__ == '__main__':

    import progressbar
    
    import config

    bar_widgets = ['Monte Carlo data: ', progressbar.Percentage(),
    ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
    ]
    progbar = progressbar.ProgressBar(widgets=bar_widgets)

    nt = config.nt
    N = config.N
    eqSteps = config.eqSteps
    mcSteps = config.mcSteps
    tmin = config.tmin
    tmax = config.tmax
    output = config.monte_carlo_output
    T = np.linspace(tmin, tmax, nt)
    i = 0
    for _ in progbar(range(nt)):
        #print('-> %i/%i'%(i+1, nt))
        state = initial_state(N)
        beta = 1./T[i]
        for j in range(eqSteps): # burn in
            mcmove(state, beta)           
        data = []
        for j in range(mcSteps):
            mcmove(state, beta)
            data.append(np.copy(state)) # important to use copy
        pickle.dump(data, open(output%(T[i], N), 'wb'))
        i += 1

