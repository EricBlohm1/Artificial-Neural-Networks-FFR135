import numpy as np
import pandas as pd

def init_w_in(N,M):
    mean = 0
    variance =0.002
    std = 1/np.sqrt(variance)
    w_in = np.random.normal(mean,std,(M,N))
    return w_in


def init_w_r(M):
    mean = 0
    variance = 2/500
    std = 1/np.sqrt(variance)
    w_r = np.random.normal(mean,std,(M,M))
    return w_r


#correct i value for w´s
#def update_reservoir_i(w_i, w_i_in, r_previous, input):
    #sum1 = 0
    #for j in range(0,len(r_previous)):
    #    sum1+= w_i[j]*r_previous[j]

    #sum2 = 0
    #for k in range(0,len(input)):
    #    sum2+= w_i_in[k]*input[k]
    
    #return np.tanh(sum1 + sum2)


def main():
    df = pd.read_csv('training-set.csv', header=None)
    x = df.to_numpy()

    test_df = pd.read_csv('test-set-9.csv',header=None)
    test = test_df.to_numpy()

    print(len(x[0]))
    ### Init ###
    #input neurons
    N = 3
    #reservoir neurons
    M = 500
    k = 0.01
    T = len(x[0])
    w_in = init_w_in(N,M)
    w_r =  init_w_r(M)
    ############

    R = np.zeros((T,M))

    #create the R matrix
    for t in range(0,T-1):
        r_next = np.zeros(M)
        r_t = R[t].copy()
        sum1= np.dot(w_r,r_t)
        sum2 =np.dot(w_in, x[:, t])

        r_next = np.tanh(sum1+sum2)
        R[t+1] = r_next

    I = np.identity(M)
    #X had each column as a time step, R has each row as a time step. Transpose x in order to make the calculation correct.
    x=x.T
    inverse = np.linalg.inv(np.dot(R.T,R) + k * I)
    dot1 = np.dot(inverse,R.T)
    w_out = np.dot(dot1,x) 
    print(w_out)

    #Calculate output
    O = np.zeros((T,N))
    for t in range(0,T-1):
        O[t+1] = np.dot(w_out.T,R[t+1])
    



if __name__ == "__main__":
    main()