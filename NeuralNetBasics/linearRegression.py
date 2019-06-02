<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt

def estimateCoeff(x,y):
    mx, my =np.mean(x), np.mean(y) #means of x and y
    n= np.size(x)
    ssXY = np.sum(x*y) -n*mx*my
    ssXX = np.sum(x*x) - n*mx*mx
    b1 = ssXY/ssXX
    b0 = my - b1*mx
    return(b0,b1)                   #calculated coeffiecents

def plot(x,y,b):
    y_predicted = b[0] + b[1]*x     #predicted y 
    plt.scatter(x, y, color = "g") 
    plt.plot(x, y_predicted, color = "r") 
    plt.show()

def main(): 
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
  
    # estimating coefficients 
    b = estimateCoeff(x, y) 
    print("b0:")
    print(b[0])
    print('\n')
    print("b[1]:")    
    print(b[1])
    plot(x, y, b) 
  
if __name__ == "__main__": 
    main() 

=======
import numpy as np
import matplotlib.pyplot as plt

def estimateCoeff(x,y):
    mx, my =np.mean(x), np.mean(y) #means of x and y
    n= np.size(x)
    ssXY = np.sum(x*y) -n*mx*my
    ssXX = np.sum(x*x) - n*mx*mx
    b1 = ssXY/ssXX
    b0 = my - b1*mx
    return(b0,b1)                   #calculated coeffiecents

def plot(x,y,b):
    y_predicted = b[0] + b[1]*x     #predicted y 
    plt.scatter(x, y, color = "g") 
    plt.plot(x, y_predicted, color = "r") 
    plt.show()

def main(): 
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
  
    # estimating coefficients 
    b = estimateCoeff(x, y) 
    print("b0:")
    print(b[0])
    print('\n')
    print("b[1]:")    
    print(b[1])
    plot(x, y, b) 
  
if __name__ == "__main__": 
    main() 

>>>>>>> weights removed
