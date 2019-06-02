<<<<<<< HEAD
import numpy as np

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoidDerivative(x):
    return x*(1-x)


def main():
    alphas = [0.001,0.01,0.1,1,10,100,1000]
    X = np.array([  [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    print("Input dataset is-")
    print(X)
    #output datasheet
    y = np.array([[0,1,1,0]]).T
    print("Output dataset is-")
    print(y)
    np.random.seed(1)
    #synapse_0=[]
    for alpha in alphas:
        print("\nTraining With Alpha-" + str(alpha))
        synapse0= 2*np.random.random((3,4)) -1       #weights
        synapse1= 2*np.random.random((4,1)) -1
        for j in range(60000):
            layer0= X                                #activation function sigmoid   
            layer1= sigmoid(np.dot(layer0,synapse0)) #forward prop using weights
            layer2= sigmoid(np.dot(layer1,synapse1))
            layer2error = layer2 - y                 #error
            
            if (j% 10000) == 0:
                print("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer2error))))
                
            layer2delta = 2*layer2error*sigmoidDerivative(layer2)
            layer1error = layer2delta.dot(synapse1.T)    
            #synapseDerivative = np.dot(layer0.T,layer1delta)
            layer1delta = 2*layer1error * sigmoidDerivative(layer1)
            synapse1 -= alpha * (layer1.T.dot(layer2delta))
            synapse0 -= alpha * (layer0.T.dot(layer1delta))
            
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
    alphas = [0.001,0.01,0.1,1,10,100,1000]
    X = np.array([  [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    print("Input dataset is-")
    print(X)
    #output datasheet
    y = np.array([[0,1,1,0]]).T
    print("Output dataset is-")
    print(y)
    np.random.seed(1)
    #synapse_0=[]
    for alpha in alphas:
        print("\nTraining With Alpha-" + str(alpha))
        synapse0= 2*np.random.random((3,4)) -1       #weights
        synapse1= 2*np.random.random((4,1)) -1
        for j in range(60000):
            layer0= X                                #activation function sigmoid   
            layer1= sigmoid(np.dot(layer0,synapse0)) #forward prop using weights
            layer2= sigmoid(np.dot(layer1,synapse1))
            layer2error = layer2 - y                 #error
            
            if (j% 10000) == 0:
                print("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer2error))))
                
            layer2delta = 2*layer2error*sigmoidDerivative(layer2)
            layer1error = layer2delta.dot(synapse1.T)    
            #synapseDerivative = np.dot(layer0.T,layer1delta)
            layer1delta = 2*layer1error * sigmoidDerivative(layer1)
            synapse1 -= alpha * (layer1.T.dot(layer2delta))
            synapse0 -= alpha * (layer0.T.dot(layer1delta))
            
        print("Output after training")
        print(layer1)    

      
if __name__ == "__main__": 
    main()
>>>>>>> weights removed
