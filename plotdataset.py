import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(model, dataset):
    
    plt.style.use('_mpl-gallery')

    # make the data
    x = np.array([])
    y = np.array([])
    idx =  np.array([])
    error = np.array([])
    prct = np.array([])
    for i in range(len(dataset.dataset)):
        x  = np.append(x, model.prediction(dataset.dataset[i][0]))
        y  = np.append(y, dataset.dataset[i][1])
        idx = np.append(idx, i)
        error= np.append(error,np.abs(x[i]-y[i]))
        prct = np.append(prct,(np.abs(x[i]-y[i])/y[i])*100)
    # plot
    print("mean = " , np.mean(error))
    print("std = " , np.std(error))
    print("var = " ,  np.var(error))
    fig, ax = plt.subplots()
    #l = np.linspace(0, 500, 100)
    ax.plot(x, x, label='Expectation', color="black")
    ax.scatter(x, y) #, s=sizes, c=colors, vmin=0, vmax=100)
    ax.set_ylabel("Ground truth [g]")
    ax.set_xlabel("Prediction [g]")
    ax.set_title("Plot")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    
    ax.scatter(idx,error)
    ax.set_ylabel("Error [g]")
    ax.set_xlabel("index ")
    ax.set_title("Plot")
    #plt.legend()
    plt.show()


    fig, ax = plt.subplots()
    
    ax.scatter(idx,prct)
    ax.set_ylabel("Error [%]")
    ax.set_xlabel("index ")
    ax.set_title("Plot")
    #plt.legend()
    plt.show()