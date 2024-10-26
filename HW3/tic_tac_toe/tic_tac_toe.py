import numpy as np
import matplotlib.pyplot as plt


def MoveAgent(board,Q,epsilon,p):
    new_board = board.copy()
    r = np.random.rand()
    pos = []
    if(r < (1-epsilon)):
        pos = BestMove(board,Q)
        #may return a list of possible positions.
        #pos = np.argwhere(board == action)
        if isinstance(pos,np.ndarray):
            #Choose a random position out of the possible ones.
            pos = pos[np.random.randint(len(pos))]
    else:
        # Find the indices where the value is 0
        zero_indices = np.argwhere(board == 0)
        # Randomly select one of the zero positions
        pos = zero_indices[np.random.randint(zero_indices.shape[0])]
    new_board[pos[0]][pos[1]] = p
    return new_board, pos

#Return highest Q-value for the state "board". 
def BestMove(board,Q):
    if(0 in board):
        b = board.tobytes()
        if (b in Q):
            if (isinstance(np.nanmax(Q[b]), np.ndarray)):
                    best_action = ChooseEqualProb(np.nanmax(Q[b]))
                    return np.argwhere(best_action == Q[b])
            return np.argwhere(np.nanmax(Q[b]) == Q[b])
    rand_pos = np.argwhere(board == 0)
    return rand_pos

#check if game over, and return which player won
def CheckGameOver(board):
    ## check if any positions are open. 
    if(0 not in board):
        return True, 0

    ## check horisontal
    for row in range(0,len(board)):
        sum = np.sum(board[row])
        if (sum == 3): 
            return True, 1
        if(sum == -3):
            return True, -1

    ## check vertical
    for col in range(0,len(board[0])):
        sum = np.sum(board[:,col])
        if (sum == 3): 
            return True, 1
        if(sum == -3):
            return True, -1

    
    ## check diagonal ##
    #p1
    if(board[0][0] == board[1][1] == board[2][2] == 1):
        return True, 1
    #p2
    if(board[0][0] == board[1][1] == board[2][2] == -1):
        return True,-1

    #p1
    if(board[0][2] == board[1][1] == board[2][0]== 1):
        return True, 1
    #p2
    if(board[0][2] == board[1][1] == board[2][0]== -1):
        return True, -1
    #######################

    return False, 0


def ChooseEqualProb(distribution): 
    r = np.random.randint(0,len(distribution))
    return distribution[r]


#terminate states whether or not to use the "max action term". Prev state refers to state at t-2
def UpdateQ(board, prev_state, action, Q, alpha, r_p, gameover):
    if board.tobytes() in Q:
        current_tmp = Q[board.tobytes()].copy()
    else:
        current_tmp = np.zeros((3,3))

    if prev_state.tobytes() in Q:
        prev_tmp = Q[prev_state.tobytes()].copy()
    else:
        prev_tmp = np.zeros((3,3))

    ## if not game over, include best action of new board. 
    if(not gameover):
        ##get best action ##
        best_pos = BestMove(board,Q)
        if(isinstance(best_pos,np.ndarray)):
            best_pos = best_pos[np.random.randint(len(best_pos))]

        prev_tmp[action[0]][action[1]] += alpha* ( r_p+ current_tmp[best_pos[0]][best_pos[1]] - prev_tmp[action[0]][action[1]])
    else:
        prev_tmp[action[0]][action[1]] += alpha* ( r_p - prev_tmp[action[0]][action[1]])

    prev_tmp = setNan(prev_state,prev_tmp)
    Q[prev_state.tobytes()] = prev_tmp
    

def setNan(board,q):
    for i in range(0,len(board)):
        for j in range(0,len(board[0])):
            if(board[i][j] !=0):
                q[i][j] = np.nan
    return q

def main():

    #### Set parameters ####
    p1 = 1 # 'X'
    p2 = -1 # 'O'

    Q_p1 = {}
    Q_p2 = {}
    epsilon = 1
    decay_rate = 0.99
    alpha = 0.1
    K = 50000 #10**4

    freq_p1 = 0
    freq_p2 = 0
    freq_draw = 0
    
    n_gameover = 0

    draw_probabilities = []
    rounds = []
    round_interval = 1000  # Calculate draw probability every 1000 rounds

    for k in range(0, K): 
        if k > 10000:
            if k % 100:
                epsilon *= decay_rate
                #print("k: ", k, "Epsilon: ", epsilon)

        board = np.zeros((3,3))
        board_states = []
        actions = []
        ## PLayer 1 always start ##
        current_p = p1

        for t in range(0,9):
            if t != 0:
                current_p *= -1

            ## Append board before we update state, the action retrieved corresponds to the "previous state" ##
            ## Think of the state as a node, and actions as the out-going edge ##
            board_states.append(board)
            action = None
            if current_p == 1:
                #action is the position of the step taken.
                board, action = MoveAgent(board, Q_p1, epsilon, current_p) 
            elif current_p == -1:
                #action is the position of the step taken.
                board, action = MoveAgent(board, Q_p2, epsilon, current_p)               
            actions.append(action)
            #####################################################

            r_p1 = 0
            r_p2 = 0
            gameOver, winner = CheckGameOver(board)
            if(gameOver):
                n_gameover +=1
                if(winner == p1):
                    r_p1 = 1
                    r_p2 = -1
                    #reward with 1 and save to correct Q-table
                    UpdateQ(board, board_states[t-2], actions[t-2], Q_p1, alpha, r_p1, gameOver)
                    #penalize with -1 and save to corretc Q-table
                    UpdateQ(board, board_states[t-1], actions[t-1], Q_p2, alpha, r_p2, gameOver)
                    #print(f"Winner: {winner}")
                    freq_p1 +=1

                elif(winner == p2):
                    r_p2 = 1
                    r_p1 = -1
                    #reward with 1 and save to correct Q-table
                    UpdateQ(board, board_states[t-2], actions[t-2], Q_p2, alpha, r_p2, gameOver)
                    #penalize with -1 and save to corretc Q-table
                    UpdateQ(board, board_states[t-1], actions[t-1], Q_p1, alpha, r_p1, gameOver)
                    #print(f"Winner: {winner}")
                    freq_p2 +=1
                else:
                    if current_p == p1:
                        UpdateQ(board, board_states[t-2], actions[t-2], Q_p1, alpha, 0, gameOver)
                        UpdateQ(board, board_states[t-1], actions[t-1], Q_p2, alpha, 0, gameOver)
                    elif current_p == p2:
                        UpdateQ(board, board_states[t-2], actions[t-2], Q_p2, alpha, 0, gameOver)
                        UpdateQ(board, board_states[t-1], actions[t-1], Q_p1, alpha, 0, gameOver)
                    freq_draw +=1
                break
                
            if t > 1: 
                if(current_p == p1):
                    UpdateQ(board, board_states[t-2], actions[t-2], Q_p1, alpha, 0, gameOver)
                elif(current_p == p2):
                    UpdateQ(board, board_states[t-2], actions[t-2], Q_p2, alpha, 0, gameOver)

            if k % round_interval == 0 and k != 0:
                # Calculate the probability of a draw
                draw_prob = freq_draw / (freq_p1 + freq_p2 + freq_draw)
                draw_probabilities.append(draw_prob)
                rounds.append(k)

    plt.figure(figsize=(10, 6))
    plt.plot(np.array(rounds) / 1000, draw_probabilities, label="Draw probability")

    plt.xlabel("Number of rounds x $10^3$")
    plt.ylabel("Probability of draw")
    plt.title("Learning to play Tic-Tac-Toe using Q-learning")
    plt.legend()
    plt.grid(True)
    plt.show()

    #print("Q: ", Q_p1)
    print(f"Frequency wins: p1 {freq_p1/K}, p2 {freq_p2/K}")
    print(f"Frequency draw:  {freq_draw/K}")
    print("sum: ", (freq_draw+freq_p1+freq_p2)/K)
    print(f"Length of dictionaries: Q1 {len(Q_p1)}, Q2 {len(Q_p2)}")

    print(" K: ", K, "n_gameover: ", n_gameover)

            

        


if __name__ == "__main__":
    main()