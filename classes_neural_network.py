import numpy as np

class layer_create:
    def __init__(self,n_inputs,n_neurons,
                  weight_regulizer_l1 = 0, weight_regulizer_l2 = 0,
                  bias_regulizer_l1 = 0, bias_regulizer_l2 = 0 ):
        
        #initializing weights
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1,n_neurons))
        
        #initializing regulization strengts
        self.weight_regulizer_l1 = weight_regulizer_l1
        self.weight_regulizer_l2 = weight_regulizer_l2
        self.bias_regulizer_l1 = bias_regulizer_l1
        self.bias_regulizer_l2 = bias_regulizer_l2


    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias
        self.inputs = inputs
    
    #backpropogation
    def backward(self,dvalues):
    
        #gradient on parameters
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis = 0, keepdims=True)

        #gradients on regulization 
        #l1 on weights
        if self.weight_regulizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0 ] = -1
            self.dweights += self.weight_regulizer_l1 * dl1

        #l2 on weights
        if self.weight_regulizer_l2 > 0:
            self.dweights += 2 * self.weight_regulizer_l2 * self.weights

        #l1 on bias
        if self.bias_regulizer_l1 > 0:
            dl1 = np.ones_like(self.bias)
            dl1[self.bias < 0] = -1
            self.dbiases += self.bias_regulizer_l1 * dl1

        #l2 on bias
        if self.bias_regulizer_l2 > 0:
            self.dbiases += 2 * self.bias_regulizer_l2 * self.bias
        
        #gradient on values
        self.dinputs = np.dot(dvalues,self.weights.T)

# Activation 

class ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
        self.inputs = inputs

    def ReLu_derivative(x):
        return np.where(x>0,1,0)
    
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let’s make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis = 1,keepdims=True))
        probablity = exp_values / np.sum(exp_values,axis = 1,keepdims=True)
        self.output = probablity

    def backward(self,y_pred,y_true):
        samples = len(y_pred)
        #turning into discrete values if hot encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis = 1)
        #copy so we can safely modify
        self.dinputs = y_pred.copy()
        #calculate gradient 
        self.dinputs[range(samples),y_true] -= 1
        #normalization
        self.dinputs = self.dinputs/samples


        
# loss and acuracy class
#common loss class
class loss:
    #regulization loss calculation
    def regulization_loss(self,layer):
        #0 by default
        regulization_loss = 0
        #l1 regulization - weights
        #calculate only when factor greater than zero
        if layer.weight_regulizer_l1 > 0:
            regulization_loss += layer.weight_regulizer_l1 * np.sum(np.abs(layer.weights))
        #l2 regulizer - weights
        if layer.weight_regulizer_l2 > 0:
            regulization_loss += layer.weight_regulizer_l2 * np.sum(layer.weights*layer.weights)
        #l1 regulization - bias
        #calculate only when factor greater than zero
        if layer.bias_regulizer_l1 > 0 :
            regulization_loss += layer.bias_regulizer_l1 * np.sum(np.abs(layer.bias))
        #l2 regulization - weights
        if layer.bias_regulizer_l2 > 0 :
            regulization_loss += layer.bias_regulizer_l2 * np.sum(layer.bias*layer.bias)
        
        return regulization_loss 
    
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

#categorical cross entropy loss class    
class loss_categorical_cross_entropy(loss):
    def forward(self,y_pred,y_true):
        n_samples = len(y_pred)
        y_pred_cliped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1 :
            correct_confidence = y_pred_cliped[range(n_samples),y_true]
        #for one-hot encoded labels
        elif len(y_true.shape) == 2 :
            correct_confidence = np.sum(y_pred_cliped * y_true,axis = 1)
        
        #losses 
        neg_log_likelihood = -np.log(correct_confidence)
        return neg_log_likelihood
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


 
class acuracy:
    def __init__(self,output,class_targets):
        predictions = np.argmax(output,axis = 1)
        if len(class_targets.shape) == 2 : 
            class_targets = np.argmax(class_targets,axis = 0)
        self.acuracy = np.mean(predictions == class_targets)

# Optimization

# SGD optimizer
class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.bias += -self.learning_rate * layer.dbiases

#SGD optimizer with learning rate decay 

class Optimizer_SGD_lrd:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.bias += -self.current_learning_rate * layer.dbiases

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_SGD_momentum:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.bias)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * layer.weight_momentums - \
                             self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - \
                           self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.bias += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
    
#combination of categorical_cross_entropy_loss and softmax activation
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = loss_categorical_cross_entropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Adagrad optimizer
class Optimizer_Adagrad:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

#optimizer root mean square propogation
class Optimizer_RMSprop:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
                           (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer
class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.968, beta_2=0.989):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.bias)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.bias += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class layer_dropout:
    def __init__(self,rate):
        #stores the dropout rate 
        self.rate = 1 - rate
    
    def forward(self,inputs):
        #save input values
        self.inputs = inputs
        #generate and save the scaled binary mask
        self.binary_mask = np.random.binomial(1,self.rate,size = inputs.shape) / self.rate
        #apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self,dvalues):
        #gradient on values
        self.dinputs = dvalues * self.binary_mask

