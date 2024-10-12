import numpy as np

def init(n, mean):
    variance = 1/n
    standard_dev = np.sqrt(variance)

    weights = np.random.normal(mean, standard_dev, size=n)
    threshold = 0

    return weights, threshold


def sample_boolean_function(n):
    tmp = []
    for _ in range(0,2**n):
        tmp.append(np.random.choice([-1, 1]))
    return tmp

#return all combinations
def create_input_matrix(n):
    matrix= []
    max = 2**n
    while len(matrix) < max:
        x_row = []
        for col in range(0,n):
            x_row.append(np.random.choice([-1, 1]))
        if(x_row not in matrix):    
            matrix.append(x_row)
    return matrix

# iterate each row in the input matrix, that same row has a correspodning t value.
# amount of patterns (mu), len(t and Output) are the same. Pattern x[0] correspond to output[0] and t[0]
# So, i am using mu to index both the patterns, t and output
def train_perceptron(weights,threshold,x,output,t,eta):
        for mu in range(0,len(x)):
            b = np.dot(weights,x[mu])
            output[mu] = np.sign(b - threshold)
            #update if output != t
            if (output[mu] != t[mu]):
                for j in range(0,len(weights)):
                    weights[j] += eta*(t[mu]-output[mu])*x[mu][j]
                threshold += -eta*(t[mu]-output[mu])
        return weights,threshold,output


#linearly separable functions for n
#n_2 = 14, n_3=104, n_4=1882, n_5 =94572
def main():
    mean = 0
    eta  = 0.05
    for n in range(2,6):
        #all possible combinations of x (imagine the left side of a boolean truth table)
        x = create_input_matrix(n)
        checked = []
        nLinearlySeparable= 0
        for _ in range(0,10**4):
                weights, threshold = init(n,mean)
                output = np.zeros(2**n)
                t = sample_boolean_function(n)
                if not (t in checked):
                    checked.append(t)
                    for epoch in range(0,20):
                        weights,threshold,output = train_perceptron(weights,threshold,x,output,t,eta)
                    if (np.array_equal(t,output)):
                        nLinearlySeparable +=1
        print("n=",n,",", "Linearly separable functions:" ,nLinearlySeparable)
        print("Number of boolean functions:" , (2**(2**n)))
        print("Fraction of linearly separable functions (fraction between nLinearlySeparable and len(checked)): ",nLinearlySeparable,"/",len(checked),"=", nLinearlySeparable/(len(checked)))
        print("\n")


                    










if __name__ == "__main__":
    main()