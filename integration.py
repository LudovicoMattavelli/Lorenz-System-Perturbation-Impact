import numpy as np
import lorenz
#Define the necessary parameters for the integration
system_parameters_A = [10,8/3,28]
time_step = 0.005
end_time = 60
total_steps = int(end_time/time_step)
initial_conditions = np.array([9, 10, 18], dtype='float')
epsilon = np.array([0, 1E-07, 1E-05, 1E-03, 1E-01])
 
istant_variable = np.ndarray([len(epsilon), 3])
X_storage = np.ndarray([len(epsilon), total_steps])
variance_storage = np.ndarray([len(epsilon), total_steps])

istant_variable = np.tile(initial_conditions, (len(epsilon), 1))
istant_variable[:, 0] += epsilon
X_storage[:, 0] = istant_variable[:, 0]
variance_storage[:,0] = 0
for i in range (total_steps-1):
    for j in range (len(epsilon)):
        istant_variable[j,:] = lorenz.euler_forward_method(istant_variable[j,:], system_parameters_A, time_step) 
        system_variance = np.power((istant_variable[0,0] - istant_variable[j,0]), 2) 
        variance_storage[j,i+1] = system_variance
        X_storage[j,i+1] = istant_variable[j,0]