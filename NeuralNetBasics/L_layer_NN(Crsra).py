<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

# Neural Network with N layers and activation functions of all hidden layers as tanh and output layer as sigmoid
np.random.seed(1)

def sigmoid(X):
    op = 1/(1+np.exp(-X))
    np.seterr(over='ignore')
    return op

def initialize_param_deep(layer_dims):
    L = len(layer_dims)
    parameters = {}                #parameters dictionary
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.random((layer_dims[l],layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    print(parameters)
    return parameters

def linear_forward(A_prev,W,b):
    Z = np.dot(W,A_prev) + b
    cache = (A_prev,W,b)         #Z -> OUTPUT OF THAT LAYER
    return Z,cache                 #cache -> Inputs and outputs of that layer

def linear_activation_forward(A_prev,W,b,activation):
    Z,L_cache = linear_forward(A_prev,W,b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "tanh":
        A = np.tanh(Z)
    A_cache = (Z,A)    
    cache = (L_cache,A_cache)
    return A,cache

def L_model_forward(X,parameters):
    L = len(parameters)//2
    caches = []
    A = X
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)],'tanh')
        caches.append(cache)

    AL,cache = linear_activation_forward(A,parameters['W' + str(L)],parameters['b' + str(L)],'sigmoid')
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y):
    m = Y.shape[1]

    cost = -1/m*(np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y)))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ,cache):
    A_prev,W,b = cache              #retrieve value from cache
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ,A_prev.T)
    db = 1/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    L_cache,A_cache = cache
    Z,A = A_cache
    if activation == "tanh":
        dZ = np.multiply(dA,(1 - np.power(A, 2)))
    elif activation == "sigmoid":
        dZ = np.multiply(dA,np.multiply(A,1-A))
        
    dA_prev,dW,db = linear_backward(dZ,L_cache)
    return dA_prev,dW,db

def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)                #list of caches (L caches)   
    m = AL.shape[1]

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]            #last cache of the neural net.Has length of L.Indexing from 0 to L-1
    grads['dA'+ str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL,current_cache,'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA'+ str(l)],grads['dW' + str(l+1)],grads['db' + str(l+1)] = linear_activation_backward(grads["dA" + str(l+1)],current_cache,'tanh')
    return grads
    
        
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):

    costs = []                         # keep track of cost
    
    parameters = initialize_param_deep(layers_dims)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X,parameters)
        print("AL: {}".format(AL))
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
             
    print(AL)
    return parameters,AL


X = np.random.random((3,3))  #3 rows = 3inputs | 3 columns = 3 training set
Y = np.array([[1,0,1]])        #3 columns = 3 outputs of 3 training sets
itr = 10
layers_dims = (3,4,1)
parameters,AL =  L_layer_model(X,Y,layers_dims,1.5,itr,False)

    
=======
import numpy as np
import matplotlib.pyplot as plt

# Neural Network with N layers and activation functions of all hidden layers as tanh and output layer as sigmoid
np.random.seed(1)

def sigmoid(X):
    op = 1/(1+np.exp(-X))
    np.seterr(over='ignore')
    return op

def initialize_param_deep(layer_dims):
    L = len(layer_dims)
    parameters = {}                #parameters dictionary
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.random((layer_dims[l],layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    print(parameters)
    return parameters

def linear_forward(A_prev,W,b):
    Z = np.dot(W,A_prev) + b
    cache = (A_prev,W,b)         #Z -> OUTPUT OF THAT LAYER
    return Z,cache                 #cache -> Inputs and outputs of that layer

def linear_activation_forward(A_prev,W,b,activation):
    Z,L_cache = linear_forward(A_prev,W,b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "tanh":
        A = np.tanh(Z)
    A_cache = (Z,A)    
    cache = (L_cache,A_cache)
    return A,cache

def L_model_forward(X,parameters):
    L = len(parameters)//2
    caches = []
    A = X
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)],'tanh')
        caches.append(cache)

    AL,cache = linear_activation_forward(A,parameters['W' + str(L)],parameters['b' + str(L)],'sigmoid')
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y):
    m = Y.shape[1]

    cost = -1/m*(np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y)))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ,cache):
    A_prev,W,b = cache              #retrieve value from cache
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ,A_prev.T)
    db = 1/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    L_cache,A_cache = cache
    Z,A = A_cache
    if activation == "tanh":
        dZ = np.multiply(dA,(1 - np.power(A, 2)))
    elif activation == "sigmoid":
        dZ = np.multiply(dA,np.multiply(A,1-A))
        
    dA_prev,dW,db = linear_backward(dZ,L_cache)
    return dA_prev,dW,db

def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches)                #list of caches (L caches)   
    m = AL.shape[1]

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]            #last cache of the neural net.Has length of L.Indexing from 0 to L-1
    grads['dA'+ str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL,current_cache,'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA'+ str(l)],grads['dW' + str(l+1)],grads['db' + str(l+1)] = linear_activation_backward(grads["dA" + str(l+1)],current_cache,'tanh')
    return grads
    
        
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):

    costs = []                         # keep track of cost
    
    parameters = initialize_param_deep(layers_dims)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X,parameters)
        print("AL: {}".format(AL))
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
             
    print(AL)
    return parameters,AL


X = np.random.random((3,3))  #3 rows = 3inputs | 3 columns = 3 training set
Y = np.array([[1,0,1]])        #3 columns = 3 outputs of 3 training sets
itr = 10
layers_dims = (3,4,1)
parameters,AL =  L_layer_model(X,Y,layers_dims,1.5,itr,False)

    
>>>>>>> weights removed
