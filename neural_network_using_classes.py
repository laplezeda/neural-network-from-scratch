#database
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
X, y = spiral_data(samples=1000, classes=3)
import classes_neural_network as cl

#create layers
layer1 = cl.layer_create(2,64,weight_regulizer_l2 = 5e-4 , bias_regulizer_l2 = 5e-4)
activation1 = cl.ReLU()
dropout1 = cl.layer_dropout(0.1)
layer2 = cl.layer_create(64,3)
activation_loss_function = cl.Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = cl.Optimizer_Adam(learning_rate= 0.02 ,decay = 1e-3)

#optimization
for i in range(10001):
    #forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    dropout1.forward(activation1.output)
    layer2.forward(dropout1.output)
    data_loss = activation_loss_function.forward(layer2.output,y)
    
    #regulization penalty
    regulization_loss = activation_loss_function.loss.regulization_loss(layer1) + \
                        activation_loss_function.loss.regulization_loss(layer2)

    loss = data_loss + regulization_loss
    
    #acuracy
    ac = cl.acuracy(activation_loss_function.output,y)
    acuracy = ac.acuracy


    #backward pass
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    
    activation_loss_function.backward(activation_loss_function.output,y)
    layer2.backward(activation_loss_function.dinputs)
    dropout1.backward(layer2.dinputs)
    activation1.backward(dropout1.dinputs)
    layer1.backward(activation1.dinputs)

    if i % 100 == 0:
        print(acuracy)
        print(loss)
        print(i)
        print(optimizer.current_learning_rate)

    #updating weights
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params

#validation

x_test,y_test = spiral_data(samples = 100 , classes = 3)
#forward pass
layer1.forward(x_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
loss = activation_loss_function.forward(layer2.output,y_test)

#acuracy
ac = cl.acuracy(activation_loss_function.output,y_test)
acuracy = ac.acuracy

print(ac.acuracy,"\n",activation_loss_function.output[:5])