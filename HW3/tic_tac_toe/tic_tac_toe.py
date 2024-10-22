import numpy as np


def MoveAgent(board,Q,epsilon,p):
    new_board = board.copy()
    r = np.random.rand()
    pos = []
    if(r < (1-epsilon)):
        action = BestMove(board,Q)
        #may return a list of possible positions.
        pos = np.argwhere(board == action)
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
                    return ChooseEqualProb(np.nanmax(Q[b]))
            return np.nanmax(Q[b])
    return 0

#check if game over, and return which player won
def CheckGameOver(board):

    ## check horisontal
    for row in range(0,len(board)):
        sum = np.sum(board[row])
        if (sum == 3): 
            return True, 1
        elif(sum == -3):
            return True, -1

    ## check vertical
    for col in range(0,len(board[0])):
        sum = np.sum(board[:,col])
        if (sum == 3): 
            return True, 1
        elif(sum == -3):
            return True, -1

    
    ## check diagonal ##
    if(board[0][0] != 0 and (board[0][0] == board[1][1] == board[2][2] == 1)):
        return True,1

    if(board[0][2] != 0 and (board[0][2] == board[1][1] == board[2][0]== -1)):
        return True,-1

    ## check if any positions are open. 
    if(0 not in board):
        #print('0 not in board')
        return True, 0

    return False,0


def ChooseEqualProb(distribution): 
    r = np.random.randint(0,len(distribution))
    return distribution[r]


#terminate states whether or not to use the "max action term". Prev state refers to state at t-2
def UpdateQ(board, prev_state, action, Q, alpha, r_p, terminate):
    return Q

def main():

    #### Set parameters ####
    p1 = 1 # 'X'
    p2 = -1 # 'O'

    Q_p1 = {}
    Q_p2 = {}
    epsilon = 1
    decay_rate = 0.9
    alpha = 0.1
    K =1 #10**4
    

    for k in range(0, K): 
        if(k % 100 == 0):
            epsilon *= decay_rate

        board = np.zeros((3,3))

        # To keep track of previous states and actions. Append the bytes of states.
        board_states = []
        actions = []

        ## PLayer 1 always start ##
        current_p = p1
        current_Q = Q_p1
        
        # one player gets one round more, ok??
        for t in range(0,9):
            ### Change between players each time step ###
            if t != 0:
                current_p *= -1
            if current_p == 1:
                current_Q = Q_p1
            elif current_p == -1:
                current_Q = Q_p2
            #############################################

            ## Append board before we update state, the action retrieved corresponds to the "previous state" ##
            ## Think of the states as a node, and actions as the out-going edge ##
            board_states.append(board.tobytes())

            #action is the position of the step taken.
            board, action = MoveAgent(board, current_Q, epsilon, current_p) 
            print(f"step:\n {t} \nboard:\n {board} \naction1:\n {action} \ncurrent player:\n {current_p}")
            print("----------------------------")

            actions.append(action)
            #####################################################

            r_p1 = 0
            r_p2 = 0
            gameOver, winner = CheckGameOver(board)
            if(gameOver):
                if(winner == p1):
                    r_p1 = 1
                    r_p2 = -1
                elif(winner == p2):
                    r_p2 = 1
                    r_p1 = -1
                print(f"Winner: {winner}")
                break
            



if __name__ == "__main__":
    main()