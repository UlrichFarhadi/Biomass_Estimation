import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def plot_dataset(model, dataset):

    plt.style.use('_mpl-gallery')

    # make the data
    x = np.array([])
    y1 = np.array([])
    y2 = np.array([])
    y3 = np.array([])
    for i in range(len(dataset.dataset)):
        x  = np.append(x, i)
        #y1  = np.append(y, np.abs(model.prediction(dataset.dataset[i][0]) - dataset.dataset[i][1]))
        y1 = np.append(y1, model.prediction(dataset.dataset[i][0]))
        y2 = np.append(y2, dataset.dataset[i][1])
        #print("i, Pred, GT: ", i, " ", model.prediction(dataset.dataset[i][0]), " ", dataset.dataset[i][1])
        y3 = np.append(y3, np.abs(model.prediction(dataset.dataset[i][0]) - dataset.dataset[i][1]))

    error = np.sum(y3) / len(dataset.dataset)
    print("Error = ", error)
    # plot
    fig, ax = plt.subplots(figsize=(12, 3))

    #ax.scatter(x, y) #, s=sizes, c=colors, vmin=0, vmax=100)
    ax.scatter(x, y1, c='g')
    ax.scatter(x, y2, c='b')
    ax.set_ylabel("Plant weight [g]")
    ax.set_xlabel("Plant [nr]")
    ax.set_title("Plant Weight prediction (green) vs GT (blue)")

    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #     ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()

    mu = np.mean(y3)
    variance = np.var(y3)
    sigma = math.sqrt(variance)
    print("mean = ", mu)
    print("variance = ", variance)
    print("Sigma = ", sigma)
    x = np.linspace(mu - 6*sigma, mu + 6*sigma, len(dataset.dataset))
    plt.plot(x, stats.norm.pdf(y3, mu, sigma))
    plt.show()