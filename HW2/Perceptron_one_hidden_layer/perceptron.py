import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data has the input in the first two elements and output on the third.
def GetInputOutput(data):
    input = []
    output = []
    for row in data:
        input.append([row[0],row[1]])
        output.append(row[2])
    return np.array(input),np.array(output)
    

def getCSV(file):
    df = pd.read_csv(file,header=None)
    data_list = df.values.tolist()
    input,output = GetInputOutput(data_list)
    return input,output


def init_w_theta(M):
    # between input and hidden
    # 1/2 from the number of inputs
    variance = 1/2
    standard_dev = np.sqrt(variance)
    theta_j = np.zeros(M)
    w_jk = np.random.normal(0, standard_dev, size=(M,2)) 

    # between hidden and output
    variance_h = 1/M
    standard_dev_h = np.sqrt(variance_h)
    theta = 0
    w_j = np.random.normal(0, standard_dev_h, size=(M)) 
    return w_jk, w_j, theta_j, theta


def compute_hidden_output(w_jk,theta_j,input): 
    b_j = np.dot(w_jk, input.T) - theta_j.T 
    return np.tanh(b_j)


def compute_network_output(w_j,theta,hidden_output,mu):
    sum = 0
    for j in range(0,len(w_j)):
        sum += w_j[j]*hidden_output[mu][j]
    B_i = sum - theta
    return np.tanh(B_i)


def back_prop(output_error,w_j,hidden_output,hidden_error):
    for m in range(0,len(w_j)):
        hidden_error[m] = output_error * w_j[m]* (1-hidden_output[m]**2)
    return hidden_error


def get_delta_w(input,hidden_output,hidden_error,output_error,eta, mini_batch,M):
    delta_w_j = np.zeros(M)
    #Delta_m, V_n. m = 1 only 1 output per pattern
    for n in range(0,len(hidden_output[0])):
        for mu in range(0,mini_batch):
            delta_w_j[n]+= output_error[mu]*hidden_output[mu][n]


    delta_w_jk = np.zeros((M,len(input[0])))
    #m is every hidden neuron.
    for m in range(0,len(hidden_error[0])):
        #n is x_1 and x_2
        for n in range(0,len(input[0])):
            for mu in range(0,mini_batch):
                delta_w_jk[m][n] += hidden_error[mu][m]*input[mu][n]

    return eta*delta_w_jk, eta*delta_w_j


def get_delta_theta(output_error,hidden_error,eta, mini_batch,M):
    delta_theta = 0
    #m = 1 
    for mu in range(0,mini_batch):
        delta_theta += output_error[mu]
    
    delta_theta_j = np.zeros(M)
    for m in range(0,len(hidden_error[0])):
        for mu in range(0,mini_batch):
            delta_theta_j[m] += hidden_error[mu][m]
    
    return -eta*delta_theta_j, -eta*delta_theta
    

def compute_classification_error(output,target):
    sum = 0
    for mu in range(0,len(target)):
        sum+= np.abs((np.sign(output[mu])-target[mu]))
    return (1/(2*len(target)))*sum


def compute_energy_function(output,target):
    sum = 0
    for mu in range(0,len(target)):
        sum += (target[mu]-output[mu])**2
    return 0.5*sum


def save_values(weights_jk,weights_j,threshold_1,threshold_2):
    df = pd.DataFrame(weights_jk)
    df.to_csv('w1.csv', index=False,header=False)

    df = pd.DataFrame(weights_j)
    df.to_csv('w2.csv',index=False, header=False)

    df = pd.DataFrame(threshold_1)
    df.to_csv('t1.csv',index=False, header=False)

    df = pd.DataFrame([threshold_2])
    df.to_csv('t2.csv',index=False, header=False)


def main():

    ##### configuration #####
    M = 10 
    epochsMax = 500
    batch_size = 64
    eta = 0.01
    #########################

    ### Retrieve data ###
    input,target = getCSV('training_set.csv')
    input_validation,target_validation = getCSV('validation_set.csv')
    ####################

    #### Center and normalize data ####
    input_mean = np.mean(input, axis=0)
    input_std = np.std(input, axis=0)
    ## Normalize based in training metrics 
    input = (input - input_mean) / input_std
    input_validation = (input_validation - input_mean) / input_std
    #########################################

    ### initialize weights and thresholds ###
    w_jk,w_j,theta_j,theta = init_w_theta(M)
    #########################################


    ## used for plotting ##
    c_train_list = np.zeros(epochsMax)
    c_validate_list = np.zeros(epochsMax)
    #######################
    
    for epoch in range(0,epochsMax):
        ### Shuffle the input data and targets ###
        indices = np.arange(len(input))
        np.random.shuffle(indices)
        input = input[indices]
        target = target[indices]
        ##########################################

        #### Create mini batches #####
        for start in range(0, len(input), batch_size):
            end = start + batch_size
            mini_batch = input[start:end]
            target_batch = target[start:end]

            ### Initialize outputs for the mini-batch ###
            hidden_output = np.zeros((len(mini_batch), M))
            output = np.zeros(len(mini_batch))
            output_error = np.zeros(len(mini_batch))
            hidden_error = np.zeros((len(mini_batch),M))

            #### for each pattern in mini batch  ####
            for mu in range(0,len(mini_batch)):
                #only one layer
                ##### Feed forward #####
                hidden_output[mu] = compute_hidden_output(w_jk,theta_j,mini_batch[mu])
                output[mu] = compute_network_output(w_j,theta,hidden_output,mu)
                ########################

                ##### back propagation #####
                output_error[mu] = (target_batch[mu]-output[mu])*(1-output[mu]**2)
                for m in range(0,len(w_j)):
                    hidden_error[mu][m] = output_error[mu] * w_j[m]* (1-hidden_output[mu][m]**2)
                ############################
                
            ##### Update weights #####
            #print(f"\n-Weights_jk before update: {w_jk}, \nWeights_j before update: {w_j}")
            delta_w_jk, delta_w_j = get_delta_w(mini_batch,hidden_output,hidden_error,output_error,eta, len(mini_batch),M)
            w_jk+= delta_w_jk
            w_j += delta_w_j

            delta_theta_j,delta_theta = get_delta_theta(output_error,hidden_error,eta, len(mini_batch),M)
            theta_j+= delta_theta_j
            theta  += delta_theta
            ##########################

        ### validate during training and early stop ###
        hidden_output_validate = np.zeros((len(input_validation),M))
        output_validate = np.zeros(len(input_validation))
        for mu in range(0,len(input_validation)):
            hidden_output_validate[mu] = compute_hidden_output(w_jk,theta_j,input_validation[mu]) 
            output_validate[mu] = compute_network_output(w_j,theta,hidden_output_validate,mu)
        ################################################

        ## Compute classification error and energy function. ##    
        c = compute_classification_error(output_validate,target_validation)
        H_validate = compute_energy_function(output_validate,target_validation)
        print("C:", c*100, ", Energy function: ", H_validate)
        c_validate_list[epoch] = c*100
        if (c < 0.12):
            break
        ################################################

    #if stopped early, retrieve all no negative elements
    c_validate_list = c_validate_list[c_validate_list > 0]

    ## create list for plotting ##
    epochs = np.arange(len(c_validate_list))
    
    ## plot Validation classification error ##
    plt.plot(epochs, c_validate_list, marker='o', color='green', markersize=4, linestyle='-')
    plt.title('Validation Classification Error')
    plt.xlabel('Epochs')
    plt.ylabel('c_validate')
    plt.grid()

    plt.tight_layout()
    plt.show()
    ########################################################

    ### Validate network after training ###
    hidden_output_validate = np.zeros((len(input_validation),M))
    output_validate = np.zeros(len(input_validation))
    for mu in range(0,len(input_validation)):
        hidden_output_validate[mu] = compute_hidden_output(w_jk,theta_j,input_validation[mu]) 
        output_validate[mu] = compute_network_output(w_j,theta,hidden_output_validate,mu)
    c = compute_classification_error(output_validate,target_validation)
    H_validate = compute_energy_function(output_validate,target_validation)
    print("C:", c*100, ", Energy function: ", H_validate)
    #######################################

    ### save the weights and thresholds to csv files ###
    #save_values(w_jk,w_j,theta_j,theta)
    ####################################################


if __name__ == "__main__":
    main()