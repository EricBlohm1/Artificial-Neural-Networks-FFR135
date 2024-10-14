import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def neighbourhood_function(r_i,r_i0,sigma):
    abs_r = np.linalg.norm(r_i - r_i0)
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

def plot_winning_neruons(ax, data, labels, w, title):
    #Make sure to only include one legend for each class and color. 
    plotted_labels = set()

    for i, x in enumerate(data):
        i0 = find_winning(w, x)
        if labels[i] == 0:  
            # Check if this label has already been plotted (for the legend)
            if 0 not in plotted_labels:
                ax.scatter(i0[0], i0[1], color='green', marker='^', s=100, alpha=0.7, label='Class 0')
                plotted_labels.add(0)
            else:
                ax.scatter(i0[0], i0[1], color='green', marker='^', s=100, alpha=0.7)

        elif labels[i] == 1:  
            if 1 not in plotted_labels:
                ax.scatter(i0[0], i0[1], color='red', marker='s', s=100, alpha=0.7, label='Class 1')
                plotted_labels.add(1)
            else:
                ax.scatter(i0[0], i0[1], color='red', marker='s', s=100, alpha=0.7)

        elif labels[i] == 2:  
            if 2 not in plotted_labels:
                ax.scatter(i0[0], i0[1], color='blue', marker='o', s=100, alpha=0.7, label='Class 2')
                plotted_labels.add(2)
            else:
                ax.scatter(i0[0], i0[1], color='blue', marker='o', s=100, alpha=0.7)

    #ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([0, 40])
    ax.set_ylim([0, 40])
    ax.set_facecolor('white')  # Set background color to white
    ax.grid(True)


def main():
    ### Retreive data and labels from CSV ###
    df = pd.read_csv('iris-data.csv', header=None)
    data = df.to_numpy()

    df = pd.read_csv('iris-labels.csv', header=None)
    labels = df.to_numpy()
    #########################################

    ### Init and parameter settings ###
    w = np.random.uniform(low=0, high=1, size=(40, 40, 4))
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

    ### Plot before weight updates ### 
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    plot_winning_neruons(axes[0], data, labels, w, title='Winning neurons before weight updates')
    ##################################


    for epoch in range(0,epochs):
        eta = eta_0*np.exp(-d_eta*epoch)
        sigma = sigma_0 * np.exp(-d_sigma*epoch)

        # p weight updates each epoch, p is the number of patterns. 
        for _ in range(0,len(data)):
            delta_w = np.zeros((40,40,4))
            idx = np.random.randint(0, len(data))
            x = data[idx]
            i0 = find_winning(w,x)
            r_i0 = np.array([i0[0], i0[1]])
            for i in range(0,w.shape[0]):
               for j in range(0,w.shape[1]):
                   r_i = np.array([i, j])
                   h = neighbourhood_function(r_i,r_i0,sigma)
                   delta_w[i][j] = eta* h *(x-w[i][j])
                
            w += delta_w
        
    ### Plot after updates ###
    plot_winning_neruons(axes[1], data, labels, w, title='Winning neurons after weight updates')

    ### Create a single legend outside the subplots ###
    handles, labels = axes[0].get_legend_handles_labels()  
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.90, 0.88)) 
    plt.tight_layout(rect=[0, 0, 0.92, 1])  
    plt.show()
    ##########################

if __name__ == "__main__":
    main()