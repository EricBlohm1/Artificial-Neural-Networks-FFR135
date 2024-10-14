import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def neighbourhood_function(r_i,r_i0,sigma):
    abs_r = np.abs(r_i-r_i0)
    frac = 1/(2*(sigma**2))
    return np.exp(-frac*(abs_r**2))


def find_winning(w,x):
    #very big initial distance
    min_dist = float('inf')
    i0 = None
    for i in range(0,w.shape[0]):
        for j in range(0,w.shape[1]):
            dist = np.linalg.norm(x-w[i][j])
            if dist < min_dist:
                min_dist = dist
                i0 = [i,j]
    return i0




def main():
    ### Retreive data and labels from CSV ###
    df = pd.read_csv('iris-data.csv', header=None)
    data = df.to_numpy()

    df = pd.read_csv('iris-labels.csv', header=None)
    labels = df.to_numpy()
    #########################################

    ### Init and parameter settings ###
    w = np.random.normal(size=((40,40,4)))
    r = np.zeros((40,40))
    eta_0 = 0.1    # initial learning rate
    d_eta = 0.001  # decay rate for eta
    sigma_0 = 10   # width of neighbourhood
    d_sigma = 0.05 # decay rate of neighbourhood
    epochs = 10
    ###################################

    ### Standardize ###
    data = data/np.max(data)
    ###################


    for epoch in range(0,epochs):
        eta = eta_0*np.exp(-d_eta*epoch)
        sigma = sigma_0 * np.exp(-d_sigma*epoch)

        # p weight updates each epoch, p is the number of patterns. 
        for _ in range(0,len(data)):
            delta_w = np.zeros((40,40,4))
            idx = np.random.randint(0, len(data))
            x = data[idx]
            i0 = find_winning(w,x)
            #
            r_i0 = r[i0[0]][i0[1]]

            r[i0[0]][i0[1]] += 1

            for i in range(0,w.shape[0]):
               for j in range(0,w.shape[1]):
                   r_i = r[i][j]
                   h = neighbourhood_function(r_i,r_i0,sigma)
                   delta_w[i][j] = eta* h *(x-w[i][j])
                   

            w += delta_w
        
    for i, x in enumerate(data): 
        i0 = find_winning(w, x) 
        if labels[i] == 0:  
            plt.scatter(r[i0[0]], r[i0[1]], color='green', marker='o', s=100, alpha=0.7)
        elif labels[i] == 1:  
            plt.scatter(r[i0[0]], r[i0[1]], color='red', marker='o', s=100, alpha=0.7)
        elif labels[i] == 2:  
            plt.scatter(r[i0[0]], r[i0[1]], color='blue', marker='o', s=100, alpha=0.7)

    plt.title('BMU Positions for Iris Dataset')
    plt.xlabel('SOM x-index')
    plt.ylabel('SOM y-index')
    plt.xlim([0, 40])
    plt.ylim([0, 40])
    plt.gca().set_facecolor('white')  # Set background color to white
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()