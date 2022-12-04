"""
Interface to call optimization solvers with a tuning procedure
using Bayesian optimization

"""

import numpy as np
from hyperopt import hp, tpe, Trials, fmin

class OPT(object):

    def __init__(self, gradient, x0):
        self.g = gradient
        self.x0 = x0
        
        self.params = {} # parameters of the solver
        self.best = {} # best parameters for the solver
        self.max_iter = 1e3
        self.tol = 1e-5

    def set_solver(self, solver):
        self.solver = solver
    
    def solve(self, **params):
        if params:
            #print("Using input parameters.")
            p = params
        elif self.params:
            #print("Using internal parameters.")
            p = self.params
        elif self.best:
            #print("Using previously tuned parameters.")
            p = self.best 
        else:
            raise ValueError('* No parameters for solver.')
        self.scores, self.states = self.solver(self.g, self.x0,  
                            max_iter=self.max_iter, tol=self.tol, **p)

    def tune(self, param_range, max_evals=100):
        
        def tune_objective(theta):
            # this function must accept a single argument (parameters)
            # and returns a score
            self.solve(**theta)
            return self.scores[-1]

        space = {}  # range for parameter search
        for par_name, par_range in param_range.items():
            space[par_name] = hp.uniform(par_name, *par_range)

        tpe_algo = tpe.suggest
        tpe_trials = Trials()
        tpe_best = fmin(fn=tune_objective, space=space, algo=tpe_algo, 
                        trials=tpe_trials, max_evals=max_evals)
        self.tpe_trials = tpe_trials
        self.best = tpe_best
            
