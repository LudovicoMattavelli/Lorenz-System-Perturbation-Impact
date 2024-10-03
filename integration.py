import numpy as np
import lorenz
import random

# Define the necessary parameters for the integration
system_parameters_A = [10,8/3,28]
time_step = 0.005
end_time = 60
total_steps = int(end_time/time_step)
initial_conditions = np.array([9, 10, 18], dtype='float')
epsilon = np.array([0, 1E-07, 1E-05, 1E-03, 1E-01])
N_ensemble = 100
tempo_ensemble = 4
distribution_inf = -0.75
distribution_sup = 0.75
random.seed(42)

# Define all the other variables right now
istant_variable = np.ndarray([len(epsilon), 3])
X_storage = np.ndarray([len(epsilon), total_steps])
variance_storage = np.ndarray([len(epsilon), total_steps])
total_steps_ensemble = int(tempo_ensemble/time_step)
epsilon_ensemble = [random.uniform(distribution_inf, distribution_sup) for i in range(N_ensemble)]
istant_ensemble_variable = np.ndarray([N_ensemble, 3])
X_ensemble_storage = np.ndarray([N_ensemble, total_steps_ensemble])
variance_ensemble_storage = np.ndarray([N_ensemble, total_steps_ensemble])
X_ensemble_storage_mean = np.zeros(total_steps_ensemble)
average_ensemble = np.zeros(total_steps_ensemble)
average_mse_ensemble = np.zeros(total_steps_ensemble)
mse_of_the_average_ensemble = np.zeros(total_steps_ensemble)

# Compute the first step out of the cycle to avoid non necessary calculations
istant_variable = np.tile(initial_conditions, (len(epsilon), 1))
istant_variable[:, 0] += epsilon
X_storage[:, 0] = istant_variable[:, 0]
variance_storage[:,0] = 0
istant_ensemble_variable = np.tile(initial_conditions, (len(epsilon_ensemble), 1))
istant_ensemble_variable[:,0] += epsilon_ensemble
X_ensemble_storage[:,0] = istant_ensemble_variable[:,0]
variance_ensemble_storage[:,0] = np.power((X_storage[0,0]-X_ensemble_storage[:,0]), 2)
average_mse_ensemble[0] = np.mean(variance_ensemble_storage[:,0])

for i in range (total_steps-1):
    for j in range (len(epsilon)):
        istant_variable[j,:] = lorenz.euler_forward_method(istant_variable[j,:], system_parameters_A, time_step) 
        variance_storage[j,i+1] = np.power((istant_variable[0,0] - istant_variable[j,0]), 2) 
        X_storage[j,i+1] = istant_variable[j,0]
    if i < (total_steps_ensemble-1):
        # This is possible because total_steps_ensemble total_steps
        for j in range (N_ensemble):
            istant_ensemble_variable[j,:] = lorenz.euler_forward_method(istant_ensemble_variable[j,:], system_parameters_A, time_step)
            variance_ensemble_storage[j,i+1] = np.power((istant_variable[0,0] - istant_ensemble_variable[j,0]), 2)
            X_ensemble_storage[j,i+1] = istant_ensemble_variable[j,0]
        # Compute of the average MSE at each time step  
        average_mse_ensemble[i+1] = np.mean(variance_ensemble_storage[:,i+1])

# Compute of the MSE of the average ensemble
average_ensemble = np.mean(X_ensemble_storage, axis=0)
mse_of_the_average_ensemble = np.power((X_storage[0, :total_steps_ensemble] - average_ensemble), 2)