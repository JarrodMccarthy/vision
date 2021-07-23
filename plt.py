import numpy as np
import matplotlib.pyplot as plt

accuracy = [0.67, 0.76, 0.89, 0.99]
error = [0.5, 0.3, 0.2, 0.1]
def plotter(accuracy, error):
    plt.figure()
    plt.scatter(np.linspace(0,len(accuracy)-1, len(accuracy)), accuracy, label='Accuracy')
    plt.legend()
    plt.show()
    #plt.hold(True)
    plt.figure()
    plt.scatter(np.linspace(0,len(error)-1, len(error)), error, label='Error')
    plt.legend()
    plt.show()
    return 

#print(np.linspace(0,len(accuracy), len(accuracy)+1))
#plotter(accuracy, error)

print(1==2)