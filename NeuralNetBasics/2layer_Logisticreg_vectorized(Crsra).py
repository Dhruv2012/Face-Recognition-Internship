<<<<<<< HEAD
import numpy as np
itr=1000000
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    np.seterr( over='ignore' )
    return output


def main():
    S=2                                            #Learning Rate
    B=0.5                                          #offset 
    X=np.array([[0,1],[0,1],[1,0],[1,0]]).T        #2 inputs(per training set) with 4 training sets
    Y=np.array([[0,0,1,1]])                        #outputs corresponding to 4 training sets 
    W=np.random.randn(1,2)                         #weights of 2 inputs
    A=np.ones((1,4))                                 #predicted op through acivation function 
    Z=np.array([[1,1,1,1]])                                 #A=sigmoid(Z)
    dZ=np.ones((1,4))                             
    dW=np.ones((1,2))
    dB=1
    A=sigmoid(Z)
    print(A)
    print(np.shape(X))
    for i in range(itr):
        Z=np.dot(W,X)+B
        A=sigmoid(Z)
        #print(A)
        dZ=A-Y
        dW=1/4*np.dot(dZ,X.T)
        dB=1/4*np.sum(dZ)
        W=W-S*dW                                    #UPDATE WEIGHTS 
        B=B-S*dB                                    #UPDATE OFFSET
    #A=A.reshape(1,4)
    print(A)        
        
if __name__ == "__main__": 
    main()    
=======
import numpy as np
itr=1000000
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    np.seterr( over='ignore' )
    return output


def main():
    S=2                                            #Learning Rate
    B=0.5                                          #offset 
    X=np.array([[0,1],[0,1],[1,0],[1,0]]).T        #2 inputs(per training set) with 4 training sets
    Y=np.array([[0,0,1,1]])                        #outputs corresponding to 4 training sets 
    W=np.random.randn(1,2)                         #weights of 2 inputs
    A=np.ones((1,4))                                 #predicted op through acivation function 
    Z=np.array([[1,1,1,1]])                                 #A=sigmoid(Z)
    dZ=np.ones((1,4))                             
    dW=np.ones((1,2))
    dB=1
    A=sigmoid(Z)
    print(A)
    print(np.shape(X))
    for i in range(itr):
        Z=np.dot(W,X)+B
        A=sigmoid(Z)
        #print(A)
        dZ=A-Y
        dW=1/4*np.dot(dZ,X.T)
        dB=1/4*np.sum(dZ)
        W=W-S*dW                                    #UPDATE WEIGHTS 
        B=B-S*dB                                    #UPDATE OFFSET
    #A=A.reshape(1,4)
    print(A)        
        
if __name__ == "__main__": 
    main()    
>>>>>>> weights removed
