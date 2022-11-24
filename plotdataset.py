import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(model, dataset):
    
    plt.style.use('_mpl-gallery')

    # make the data
    x = np.array([])
    y = np.array([])
    for i in range(len(dataset.dataset)):
        x  = np.append(x, model.prediction(dataset.dataset[i][0]))
        y  = np.append(y, dataset.dataset[i][1])

    # plot
    fig, ax = plt.subplots()

    ax.scatter(x, y) #, s=sizes, c=colors, vmin=0, vmax=100)
    ax.set_ylabel("Ground truth [g]")
    ax.set_xlabel("Prediction [g]")
    ax.set_title("Plot")

    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #     ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()