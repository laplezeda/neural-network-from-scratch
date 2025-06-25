

import numpy as np

inputs = [[1,2,3],  # batch 1 
          [1,2,3],  # batch 2
          [1,2,3]]  # batch 3

# hidden layer
weight1 = [[0.1,0.2,0.3],   # weight of neuron 1  
           [0.4,0.5,0.6],   # weight of neuron 2
           [0.7,0.8,0.9]]   # weight of neuron 3

bias1 = [1,2,3]

# output
weight2 = [[0.1,0.2,0.3],   # weight of neuron 1  
           [0.4,0.5,0.6],   # weight of neuron 2
           [0.7,0.8,0.9]]   # weight of neuron 3

bias2 = [1,2,3]

# convert to numpy array
inputs_array = np.array(inputs)
weight1_array_transpose = np.array(weight1).T
bias1_array = np.array(bias1)
weight2_array_transpose = np.array(weight2).T
bias2_array = np.array(bias2)


# hidden layer output
output1 = np.dot(inputs_array,weight1_array_transpose) + bias1_array

# final output
output2 = np.dot(output1,weight2_array_transpose) + bias2_array

print(output2)

    

