import classes_neural_network as cl
import numpy as np
import nnfs
# this is for neural_network file
layers = [1,2,3,4]
from nnfs.datasets import spiral_data
nnfs.init
# creating database
X, y = spiral_data(samples=100, classes=3)

#try1
# optimization

for i in range (1,len(layers)-1):
    weights = layers[i].weights
    bias = layers[i].bias
    if i == 0:
        inputs = X
    elif i > 0 :
        inputs = layers[i-1].output
    target_output = 0.0
    learning_rate = 0.001

    #training 
    for j in range(200):
        #forward pass
        # output of a layer
        linear_output = layers[i].output

        #output after applyig relu
        relu = cl.ReLU()
        relu.forward(linear_output)
        output = relu.output

        #output after applying sum 
        y_output = np.sum(output) 

        #loss
        loss = (y_output)**2

        #backward pass
        #gradient of loss wrt y
        dloss_dy = 2*(y_output)

        #gradient of y wrt output(a)
        dy_doutput = np.ones_like(output)

        #gradient of loss wrt output(a)
        dloss_doutput = dloss_dy * dy_doutput

        #gradient of output(a) wrt linear output(z) 
        doutput_dlinear = cl.ReLU.ReLu_derivative(linear_output)

        #gradient of loss wrt linear output(z)
        dloss_dlinear = dloss_doutput * doutput_dlinear

        #gradient of linear output(z ) wrt weights 
        dlinear_dweights = np.outer(dloss_dlinear,inputs)

        #gradient of linear output(z) wrt bias
        dlinear_dbias = dloss_dlinear

        #update weights
        weights -= learning_rate * dlinear_dweights
        bias -= learning_rate * dlinear_dbias
        
        #printing iteration and loss
        print(f"itteration {i+1}, loss : {loss}")
    
    
    print("final weights for: ", layers[i]," are : ", weights)
    print("final bias for: ", layers[i]," are : ", bias)
print(layer1.output.shape)
print(activation1.output.shape)
z = np.sum(activation1.output,axis = 1 ,keepdims= True)
print(z.shape)
l = z**2
print(l.shape)





#try 2
# optimization

for i in range (1,len(layers)-1):
    weights = layers[i].weights
    bias = layers[i].bias
    if i == 0:
        inputs = X
    elif i > 0 :
        inputs = layers[i-1].output
    target_output = 0.0
    learning_rate = 0.001

    #training 
    for j in range(200):
        #forward pass
        # output of a layer
        linear_output = layers[i].output

        #output after applyig relu
        relu = cl.ReLU()
        relu.forward(linear_output)
        output = relu.output

        #output after applying sum 
        y_output = np.sum(output,axis = 1,keepdims=True) 
        print(y_output.shape)

        #loss
        loss = np.mean((y_output)**2)

        #backward pass
        #gradient of loss wrt y
        dloss_dy = 2*(y_output)

        #gradient of y wrt output(a)
        dy_doutput = np.ones_like(output)
        
        #gradient of loss wrt output(a)
        dloss_doutput = dloss_dy * dy_doutput
        
        #gradient of output(a) wrt linear output(z) 
        doutput_dlinear = cl.ReLU.ReLu_derivative(linear_output)
        
        #gradient of loss wrt linear output(z)
        dloss_dlinear = dloss_doutput * doutput_dlinear
        print(dloss_dlinear.shape)
        #gradient of linear output(z ) wrt weights 
        dlinear_dweights = np.dot(dloss_dlinear,np.sum(inputs,axis = 1,keepdims=True).T )
        print(dloss_dlinear.shape)

        #gradient of linear output(z) wrt bias
        dlinear_dbias = dloss_dlinear

        #update weights
        weights -= learning_rate * dlinear_dweights
        bias -= learning_rate * dlinear_dbias
        
        #printing iteration and loss
        print(f"itteration {i+1}, loss : {loss}")
    
    
    print("final weights for: ", layers[i]," are : ", weights)
    print("final bias for: ", layers[i]," are : ", bias)