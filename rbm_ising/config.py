import numpy as np


# data for Monte Carlo simulation
monte_carlo_output = 'data/temp_%.4f_size_%i.pickle'
nt = 25             # number temperature points
N = 8               # size of grid, N x N
eqSteps = 512       # number of burn in iterations
mcSteps = 100000    # number monte carlo steps
tmin = 0.5
tmax = 4.5
temperatures = np.linspace(tmin, tmax, nt)
mc_save_plot = 'data/mc_energy_heat.pickle'

# RBM with gradient descent
# save this one since it took more than 2 hours to finish
# make sure to rename the files below
#gd_file_format = 'rbm_samples/gd_temp_%.4f_size_%i.pickle'
#gd_conv_format = 'rbm_samples/convergence_grad_temp_%.4f_size_%i.pickle'
#gd_num_train_points = 10000
#gd_skip = 1
#gd_n_hidden = 100
#gd_batch_size = 100
#gd_learning_rate = 1e-3
#gd_iterations = 4000
#gd_save_plot = 'data/gd_energy_heat.pickle'

# RBM with gradient descent
#gd_file_format = 'rbm_samples/gd2_temp_%.4f_size_%i.pickle'
gd_file_format = 'rbm_samples/gd2_temp_%.4f_size_%i_v2.pickle'
#gd_conv_format = 'rbm_samples/convergence_grad2_temp_%.4f_size_%i.pickle'
gd_conv_format = 'rbm_samples/convergence_grad2_temp_%.4f_size_%i_v2.pickle'
gd_num_train_points = 2000
gd_skip = 1
gd_n_hidden = 400
gd_batch_size = 100
gd_learning_rate = 5e-3
gd_iterations = 2000
#gd_save_plot = 'data/gd2_energy_heat.pickle'
gd_save_plot = 'data/gd2_energy_heat_v2.pickle'

# these parameters worked quite well
# RBM with leapfrog
lf_file_format = 'rbm_samples/leap_temp_%.4f_size_%i_v2.pickle'
lf_conv_format = 'rbm_samples/convergence_leap_temp_%.4f_size_%i_v2.pickle'
lf_num_train_points = 2000
lf_skip = 1
lf_n_hidden = 400
lf_batch_size = 100
lf_learning_rate = 2.5e-2
lf_mu = 0.98
lf_iterations = 2000
lf_save_plot = 'data/leap_energy_heat_v2.pickle'

# RBM with relativistic leapfrog
rgd_file_format = 'rbm_samples/rgd_temp_%.4f_size_%i_v2.pickle'
rgd_conv_format = 'rbm_samples/convergence_rgd_temp_%.4f_size_%i_v2.pickle'
rgd_num_train_points = 2000
rgd_skip = 1
rgd_n_hidden = 400
rgd_batch_size = 100
rgd_learning_rate = 2.8e-2
rgd_mu = 0.99
rgd_c = 5
rgd_iterations = 2000
rgd_save_plot = 'data/rgd_energy_heat_v2.pickle'

# RBM with nesterov 
nest_file_format = 'rbm_samples/nest_temp_%.4f_size_%i.pickle'
nest_conv_format = 'rbm_samples/convergence_nest_temp_%.4f_size_%i.pickle'
nest_num_train_points = 2000
nest_skip = 1
nest_n_hidden = 400
nest_batch_size = 100
nest_learning_rate = 1e-2
nest_mu = 0.99
nest_iterations = 1000
nest_save_plot = 'data/nest_energy_heat.pickle'
