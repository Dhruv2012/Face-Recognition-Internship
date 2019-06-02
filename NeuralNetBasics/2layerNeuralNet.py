<<<<<<< HEAD
import numpy as np

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoidDerivative(x):
    return x*(1-x)

def main():
    X = np.array([  [0,1],[0,1],[1,0],[1,0] ])
    print("Input dataset is-")
    print(X)
    #output datasheet
    y = np.array([[0,0,1,1]]).T
    print("Output dataset is-")
    print(y)
    np.random.seed(1)
    #synapse_0=[]
    synapse0= 2*np.random.random((2,1)) -1       #weights
    for i in range(10000):
        layer0= X                                #activation function sigmoid   
        layer1= sigmoid(np.dot(layer0,synapse0)) #forward prop using weights
        layer1error = layer1 - y                 #error

        layer1delta = 2*layer1error*sigmoidDerivative(layer1)
        synapseDerivative = np.dot(layer0.T,layer1delta)
        synapse0 -= synapseDerivative             #update weights
        
    print("Output after training")
    print(layer1)    

      
if __name__ == "__main__": 
    main()
=======
import numpy as np

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoidDerivative(x):
    return x*(1-x)

def main():
    X = np.array([  [0,1],[0,1],[1,0],[1,0] ])
    print("Input dataset is-")
    print(X)
    #output datasheet
    y = np.array([[0,0,1,1]]).T
    print("Output dataset is-")
    print(y)
    np.random.seed(1)
    #synapse_0=[]
    synapse0= 2*np.random.random((2,1)) -1       #weights
    for i in range(10000):
        layer0= X                                #activation function sigmoid   
        layer1= sigmoid(np.dot(layer0,synapse0)) #forward prop using weights
        layer1error = layer1 - y                 #error

        layer1delta = 2*layer1error*sigmoidDerivative(layer1)
        synapseDerivative = np.dot(layer0.T,layer1delta)
        synapse0 -= synapseDerivative             #update weights
        
    print("Output after training")
    print(layer1)    

      
if __name__ == "__main__": 
    main()
>>>>>>> weights removed
