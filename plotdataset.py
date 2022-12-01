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

def plot_full_dataset(model, dataset):
    
    plt.style.use('_mpl-gallery')

    # make the data
    x = np.array([])
    y = np.array([])
    idex =  np.array([])
    error = np.array([])
    prct = np.array([])
    fresh = np.array([])
    # s = [2, 3, 1, 4, 5, 3]
    # sorted(range(len(s)), key=lambda k: s[k])
    temp = []
    for i in range(len(dataset.dataset)):
        temp.append(dataset.dataset[i][1])
    idx = sorted(range(len(temp)), key=lambda k: temp[k])
    for i in range(len(dataset.dataset)):
        pred = model.prediction(dataset.dataset[idx[i]][0])
        true_val = dataset.dataset[idx[i]][1]
        x  = np.append(x, pred)
        y  = np.append(y, true_val)
        idex = np.append(idex, i)
        error= np.append(error,np.abs(pred-true_val))
        prct = np.append(prct,(np.abs(pred-true_val)/true_val)*100)
        #print(true_val.numpy()[0])
       # fresh = np.append(fresh,abs(pred.numpy()[0][0]-true_val.numpy()[0]))
    # plot
    #fresh_weight_GT[i], dry_weight_GT[i], height_GT[i], diameter_GT[i]
    # error = error.reshape(-1,4)
    # print(error.shape)
    # print(len(fresh))
    # goal = 0 
    # for i in range((len(fresh))):
    #     if(error[i,0] == fresh[i]):
    #         goal +=1
    
    #print(goal)
    print("diameter mean = " , np.mean(error[:]))
    print("diameter std = " , np.std(error[:]))
    print("diameter var = " ,  np.var(error[:]))

    # print("dry weight mean = " , np.mean(error[:,1]))
    # print("dry weight std = " , np.std(error[:,1]))
    # print("dry weight var = " ,  np.var(error[:,1]))

    # print("height mean = " , np.mean(error[:,2]))
    # print("height std = " , np.std(error[:,2]))
    # print("height var = " ,  np.var(error[:,2]))

    # print("diameter mean = " , np.mean(error[:,3]))
    # print("diameter std = " , np.std(error[:,3]))
    # print("diameter var = " ,  np.var(error[:,3]))




    fig, ax = plt.subplots()
    #l = np.linspace(0, 500, 100)
    print(x.shape)
    print(y.shape)
    ax.plot(x, x, label='Expectation', color="black")
    ax.scatter(x, y) #, s=sizes, c=colors, vmin=0, vmax=100)
    ax.set_ylabel("Ground truth [g]")
    ax.set_xlabel("Prediction [g]")
    ax.set_title("Plot")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    
    ax.scatter(idex,error)
    ax.set_ylabel("Error [g]")
    ax.set_xlabel("index ")
    ax.set_title("Plot")
    #plt.legend()
    plt.show()


    fig, ax = plt.subplots()
    
    ax.scatter(idex,prct)
    ax.set_ylabel("Error [%]")
    ax.set_xlabel("index ")
    ax.set_title("Plot")
    #plt.legend()
    plt.show()