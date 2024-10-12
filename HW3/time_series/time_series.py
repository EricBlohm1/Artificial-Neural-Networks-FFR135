import numpy as np
import pandas as pd

def init_w_in(N,M):
    mean = 0
    variance =0.002
    std = np.sqrt(variance)
    w_in = np.random.normal(mean,std,(M,N))
    return w_in


def init_w_r(M):
    mean = 0
    variance = 2/500
    std = np.sqrt(variance)
    w_r = np.random.normal(mean,std,(M,M))
    return w_r


def main():
    #### Retrieve data ####
    df = pd.read_csv('training-set.csv', header=None)
    x = df.to_numpy()

    test_df = pd.read_csv('test-set-9.csv',header=None)
    test_x = test_df.to_numpy()
    #######################

    ### Init ###
    #input neurons
    N = 3
    #reservoir neurons
    M = 500
    k = 0.01
    T = len(x[0])
    w_in = init_w_in(N,M)
    w_r =  init_w_r(M)
    predict_T = 500
    ############

    #### Calculate the reservoir using the input ####
    R = np.zeros((T,M))
    for t in range(0,T-1):
        r_next = np.zeros(M)
        r_t = R[t].copy()
        sum1= np.dot(w_r,r_t)
        sum2 =np.dot(w_in, x[:, t])

        r_next = np.tanh(sum1+sum2)
        R[t+1] = r_next
    #################################################

    #### Compute the output weights using ridge regression ###
    I = np.identity(M)
    #X had each column as a time step, R has each row as a time step. Transpose x in order to make the calculation correct.
    x=x.T
    inverse = np.linalg.inv(np.dot(R.T,R) + k * I)
    dot1 = np.dot(inverse,R.T)
    w_out = np.dot(dot1,x) 
    ##########################################################

    #### Calculate new R using test data and then using the output. ####
    R_predict = np.zeros((len(test_x[0])+predict_T,M))
    O_predict = np.zeros((len(test_x[0])+predict_T,N))

    # First create the R matrix and output using the 100 values from the test data
    # then continue to predict these values from the test data
    for t in range(0,len(test_x[0])+predict_T-1):
        r_next = np.zeros(M)
        r_t = R_predict[t].copy()
        if t < 100:
            sum1= np.dot(w_r,r_t)
            sum2 =np.dot(w_in, test_x[:, t])
            r_next = np.tanh(sum1+sum2)
        else: 
            sum1= np.dot(w_r,r_t)
            sum2 =np.dot(w_in, O_predict[t])
            r_next = np.tanh(sum1+sum2)

        R_predict[t+1] = r_next
        O_predict[t+1] = np.dot(w_out.T,R_predict[t+1])
    #######################################################################

    ### Save answer, y-component of the output ###
    O_predict = O_predict.T
    # slice off the 100 first elements, coming from the test data, not prediction.
    O_predict = O_predict[:, 100:]
    print(O_predict)
    answer = O_predict[1]
    df = pd.DataFrame([answer])
    df.to_csv('prediction.csv', index=False, header=False)
    ##############################################




if __name__ == "__main__":
    main()