
import numpy as np
import matplotlib.pyplot as plt

def plotter(accuracy):
    plt.figure()
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.scatter(np.linspace(0,len(accuracy)-1, len(accuracy)), accuracy, label='Accuracy')
    plt.legend()
    plt.show()
    #plt.hold(True)
    # plt.figure()
    # plt.scatter(np.linspace(0,len(error)-1, len(error)), error, label='Error')
    # plt.legend()
    # plt.show()
    return 


test_accuracy = [0.5, 0.65, 0.72, 0.85, 0.92, 0.95, 0.97, 0.98, 0.98, 0.99]

plotter(test_accuracy)
#plotter(train_accuracy, testerror)