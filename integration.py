import numpy as np
import lorenz
#Define the necessary parameters for the integration
system_parameters_A = [10,8/3,28]
time_step = 0.005
end_time = 60
total_steps = int(end_time/time_step)
system_variance = 0
initial_conditions = [9, 10, 18]
epsilon = np.array([0, 1E-07, 1E-05, 1E-03, 1E-01])
 
istant_variable = np.ndarray([len(epsilon), 3])
X_storage = np.ndarray([len(epsilon), total_steps])
Var_p = np.ndarray([len(epsilon), total_steps])
#Define two variables to make less expensive the cycle
Flag_0 = [True, True, True, True, True] 

for i in range (total_steps-1):
    for j in range (len(epsilon)):
        if Flag_0[j] == True : 
            istant_variable[j,:] = initial_conditions
            istant_variable[j,0] += epsilon[j]
            X_storage[j,i] = istant_variable[j,0]
            Var_p[j,i] = system_variance
            Flag_0[j] = False #Because it has to be done only the first time
        istant_variable[j,:] = lorenz.euler_forward_method(istant_variable[j,:], system_parameters_A, time_step) 
        system_variance = np.power((istant_variable[0,0] - istant_variable[j,0]), 2) 
        Var_p[j,i+1] = system_variance
        X_storage[j,i+1] = istant_variable[j,0]