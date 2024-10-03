import numpy as np
import lorenz
import random

# Define the necessary parameters for the integration
system_parameters_A = [10,8/3,28]
time_step = 0.005
end_time = 60
initial_conditions = np.array([9, 10, 18], dtype='float')
epsilon = np.array([0, 1E-07, 1E-05, 1E-03, 1E-01])
presence_of_perturbation = np.full(len(epsilon), True)
presence_of_perturbation[0] = False
N_ensemble = 100
end_ensemble_time = 4
distribution_inf = -0.75
distribution_sup = 0.75
random.seed(42)

# Define the arrays that will be necessary to store the interesting values
total_steps = int(end_time/time_step) 
istant_variable = np.tile(initial_conditions, (len(epsilon), 1))
X_storage = np.ndarray([len(epsilon), total_steps])
variance_storage = np.ndarray([len(epsilon), total_steps])
total_steps_ensemble = int(end_ensemble_time/time_step)
epsilon_ensemble = [random.uniform(distribution_inf, distribution_sup) for i in range(N_ensemble)]
istant_ensemble_variable = np.tile(initial_conditions, (N_ensemble, 1))
X_ensemble_storage = np.ndarray([N_ensemble, total_steps_ensemble])
variance_ensemble_storage = np.ndarray([N_ensemble, total_steps_ensemble])
X_ensemble_storage_mean = np.zeros(total_steps_ensemble)
average_ensemble = np.zeros(total_steps_ensemble)
average_mse_ensemble = np.zeros(total_steps_ensemble)
mse_of_the_average_ensemble = np.zeros(total_steps_ensemble)

# Compute the first step out of the cycle to avoid non necessary calculations
istant_variable[:, 0] += epsilon
X_storage[:, 0] = istant_variable[:, 0]
variance_storage[:,0] = 0
istant_ensemble_variable[:,0] += epsilon_ensemble
X_ensemble_storage[:,0] = istant_ensemble_variable[:,0]
variance_ensemble_storage[:,0] = np.power((X_storage[0,0]-X_ensemble_storage[:,0]), 2)
average_mse_ensemble[0] = np.mean(variance_ensemble_storage[:,0])

for i in range (1,total_steps):
    for j in range (len(epsilon)):            
        istant_variable[j,:] = lorenz.euler_forward_method(istant_variable[j,:], system_parameters_A, time_step)
        X_storage[j,i] = istant_variable[j,0]
        if presence_of_perturbation[j] == True:
            variance_storage[j,i] = np.power((X_storage[0,i] - X_storage[j,i]), 2)           
    if i < (total_steps_ensemble):
        for j in range (1,N_ensemble):
            istant_ensemble_variable[j,:] = lorenz.euler_forward_method(istant_ensemble_variable[j,:], system_parameters_A, time_step)
            X_ensemble_storage[j,i] = istant_ensemble_variable[j,0]
            variance_ensemble_storage[j,i] = np.power((X_storage[0,i] - X_ensemble_storage[j,i]), 2)
        average_mse_ensemble[i] = np.mean(variance_ensemble_storage[:,i])


# Compute of the MSE of the average ensemble
average_ensemble = np.mean(X_ensemble_storage, axis=0)
mse_of_the_average_ensemble = np.power((X_storage[0, :total_steps_ensemble] - average_ensemble), 2)