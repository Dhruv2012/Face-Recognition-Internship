<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

X = np.random.random((3,3))  #3 rows = 3inputs | 3 columns = 3 training set
Y = np.array([[1,0,1]])        #3 columns = 3 outputs of 3 training sets

#print(X)
#print(Y)
itr = 200

def sigmoid(X):
    op = 1/(1+np.exp(-X))
    np.seterr(over='ignore')
    return op

def layer_size(X,Y):
    shape_X = np.shape(X)
    shape_Y = np.shape(Y)
    n_x = shape_X[0]
    n_h = 4
    n_y = shape_Y[0]                            #n_x = input layer size
    m = shape_X[1]                              #n_h = hidden layer size
    return(n_x,n_h,n_y,m)                       #n_y = output layer size    

def initialize_parameters(n_x,n_h,n_y):
    W1= np.random.random((n_h,n_x))
    b1= np.zeros((n_h,1))
    W2= np.random.random((n_y,n_h))
    b2= np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,                     #parameters initialized : weights and offsets    
                  "b2": b2}
    return parameters

def forward_prop(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)                            #outputs of hidden and output layer

    return Z1,A1,Z2,A2



def backward_prop(m,parameters,X,Y,Z1,A1,Z2,A2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dZ2 = A2 - Y
    dW2 = 1/m*(np.dot(dZ2,A1.T))
    db2 = 1/m*(np.sum(dZ2,axis=1,keepdims = True))
    dZ1 = np.multiply(np.dot(W2.T,dZ2),(1 - np.power(A1, 2)))
    dW1 = 1/m*(np.dot(dZ1,X.T))
    db1 = 1/m*(np.sum(dZ1,axis=1,keepdims = True))
    gradc = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
            }
    return gradc                              #gradients 

def update_param(gradc,parameters,learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradc["dW1"]
    db1 = gradc["db1"]
    dW2 = gradc["dW2"]
    db2 = gradc["db2"]

    W1 = W1 -learning_rate*dW1
    b1 = b1 -learning_rate*db1
    W2 = W2 -learning_rate*dW2
    b2 = b2 -learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,                     #parameters update through gradient descent
                  "b2": b2}
    return parameters

def main():
    X = np.random.random((3,3))  #3 rows = 3inputs | 3 columns = 3 training set
    Y = np.array([[1,0,1]])        #3 columns = 3 outputs of 3 training sets
    itr = 200
    n_x,n_h,n_y,m = layer_size(X,Y)
    parameters = initialize_parameters(n_x,n_h,n_y)

    for i in range(itr+200):    
        Z1,A1,Z2,A2 = forward_prop(X,parameters)
        gradc = backward_prop(m,parameters,X,Y,Z1,A1,Z2,A2)
        parameters = update_param(gradc,parameters,learning_rate=1.5)
        
    print(A2)
    plt.plot(A2)
    
if __name__ == "__main__": 
    main()




=======
import numpy as np
import matplotlib.pyplot as plt

X = np.random.random((3,3))  #3 rows = 3inputs | 3 columns = 3 training set
Y = np.array([[1,0,1]])        #3 columns = 3 outputs of 3 training sets

#print(X)
#print(Y)
itr = 200

def sigmoid(X):
    op = 1/(1+np.exp(-X))
    np.seterr(over='ignore')
    return op

def layer_size(X,Y):
    shape_X = np.shape(X)
    shape_Y = np.shape(Y)
    n_x = shape_X[0]
    n_h = 4
    n_y = shape_Y[0]                            #n_x = input layer size
    m = shape_X[1]                              #n_h = hidden layer size
    return(n_x,n_h,n_y,m)                       #n_y = output layer size    

def initialize_parameters(n_x,n_h,n_y):
    W1= np.random.random((n_h,n_x))
    b1= np.zeros((n_h,1))
    W2= np.random.random((n_y,n_h))
    b2= np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,                     #parameters initialized : weights and offsets    
                  "b2": b2}
    return parameters

def forward_prop(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)                            #outputs of hidden and output layer

    return Z1,A1,Z2,A2



def backward_prop(m,parameters,X,Y,Z1,A1,Z2,A2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dZ2 = A2 - Y
    dW2 = 1/m*(np.dot(dZ2,A1.T))
    db2 = 1/m*(np.sum(dZ2,axis=1,keepdims = True))
    dZ1 = np.multiply(np.dot(W2.T,dZ2),(1 - np.power(A1, 2)))
    dW1 = 1/m*(np.dot(dZ1,X.T))
    db1 = 1/m*(np.sum(dZ1,axis=1,keepdims = True))
    gradc = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
            }
    return gradc                              #gradients 

def update_param(gradc,parameters,learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradc["dW1"]
    db1 = gradc["db1"]
    dW2 = gradc["dW2"]
    db2 = gradc["db2"]

    W1 = W1 -learning_rate*dW1
    b1 = b1 -learning_rate*db1
    W2 = W2 -learning_rate*dW2
    b2 = b2 -learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,                     #parameters update through gradient descent
                  "b2": b2}
    return parameters

def main():
    X = np.random.random((3,3))  #3 rows = 3inputs | 3 columns = 3 training set
    Y = np.array([[1,0,1]])        #3 columns = 3 outputs of 3 training sets
    itr = 200
    n_x,n_h,n_y,m = layer_size(X,Y)
    parameters = initialize_parameters(n_x,n_h,n_y)

    for i in range(itr+200):    
        Z1,A1,Z2,A2 = forward_prop(X,parameters)
        gradc = backward_prop(m,parameters,X,Y,Z1,A1,Z2,A2)
        parameters = update_param(gradc,parameters,learning_rate=1.5)
        
    print(A2)
    plt.plot(A2)
    
if __name__ == "__main__": 
    main()




>>>>>>> weights removed
