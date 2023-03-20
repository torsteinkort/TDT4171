import numpy as np

O_umbrella = np.matrix([[0.9, 0],[0, 0.2]])
O_not_umbrella = np.matrix([[0.1, 0],[0, 0.8]])
T = np.matrix([[0.7, 0.3], [0.3, 0.7]])

def forward(observation_list): # returns state probabilities of last observation
    f = np.matrix([[0.5, 0.5], [0.5, 0.5]]) # initial value based on no observations
    for umbrella_observed in observation_list:
        O = O_umbrella if umbrella_observed else O_not_umbrella 
        f = O.dot(np.transpose(T)) * f

    f = f/f.sum(axis=0) # normalize
    
    return f


print('Task 2.1: \n')
print(forward([True, True]))

print('\n\nTask 2.2: \n')
print(forward([True, True, False, True, True]))