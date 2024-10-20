import numpy as np


def MoveAgent(board, Q ,epsilon,p):
    new_board = board.copy()
    r = np.random.rand()
    pos = []
    action = None
    if(r < (1-epsilon)):
        action = BestMove(new_board,Q)
        pos = np.where(new_board == action)

    else:
        # Find the indices where the value is 0
        zero_indices = np.argwhere(new_board == 0)
        # Randomly select one of the zero positions
        pos = zero_indices[np.random.randint(zero_indices.shape[0])]
        b = new_board.tobytes()
        if b in Q:
            action = Q[b][pos[0]][pos[1]]

    new_board[pos[0]][pos[1]] = p
    return new_board, pos, action

#Return highest Q-value for the state "board". 
def BestMove(board,Q):
    if(0 in board):
        b = board.tobytes()
        if (b in Q):
            if (isinstance(np.nanmax(Q[b]), np.ndarray)):
                    return ChooseEqualProb(np.nanmax(Q[b]))
            return np.nanmax(Q[b])
    return 0

def CheckGameOver(board):

    ## check if any positions are open. 
    if(0 not in board):
        #print('0 not in board')
        return True, 0

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

    return False,0

def InitQ(Q,board):
        q = np.zeros((3,3))
        if(len(Q) == 0):
            Q[board.tobytes()] = q
        else:
            q = Q[board.tobytes()].copy()
        return Q , q

def ChooseEqualProb(distribution): 
    r = np.random.randint(0,len(distribution))
    return distribution[r]

def setNan(board,q):
    for i in range(0,3):
        for j in range(0,3):
            if board[i][j] != 0:
                q[i][j] = np.nan

def UpdateQ(board,prev_state,action1_pos, Q,alpha,r_p):
    print("prev: \n", prev_state, "\nnew: \n", board)
    action = BestMove(board,Q)

    # Q table
    
    if (board.tobytes() in Q) and (prev_state.tobytes() in Q):
        var = Q[prev_state.tobytes()]
        bestPos = np.argwhere(Q[board.tobytes()] == action)
        if (isinstance(bestPos, np.ndarray)):
            bestPos = ChooseEqualProb(bestPos)
        
        var[action1_pos[0]][action1_pos[1]] += alpha*(r_p+var[bestPos[0]][bestPos[1]]-var[action1_pos[0]][action1_pos[1]])
        var = setNan(prev_state,var)
        Q[prev_state] = var
        print("var: ", var)
    return  Q

def main():

    #### Set parameters ####
    p1 = 1 # 'X'
    p2 = -1 # 'O'

    Q_p1 = {np.zeros((3,3)).tobytes() : np.zeros((3,3))}
    Q_p2 = {np.zeros((3,3)).tobytes() : np.zeros((3,3))}
    epsilon = 1
    decay_rate = 0.9
    alpha = 0.1
    K =10 #10**4
    

    for k in range(0, K): 
        if(k % 100 == 0):
            epsilon *= decay_rate
        board = np.zeros((3,3))

        #Q_p1, q1 = InitQ(Q_p1,board)
        #Q_p2, q2 = InitQ(Q_p2,board)

        # both players put 1 piece, and they should have the same number of moves. thus range 0,4
        for t in range(0,5):
            #the state before both players made a move
            prev_state = board.copy()
            print("t",t)

            ### Player 1 ###
            board, current_pos1, action1 = MoveAgent(board, Q_p1,epsilon,p1) 
            #print(f"new board:\n {new_board}, \nq1:\n {q1}")
            r_p1 = 0
            gameOver, winner = CheckGameOver(board)
            if(gameOver):
                if(winner == p1):
                    r_p1 = 1
                elif(winner == p2):
                    r_p1 = -1
                #new position, may be a matrix with multiple points
                Q_p1 = UpdateQ(board,prev_state,current_pos1,Q_p1,alpha,r_p1)
                break
            
            ## Player 2 ##
            board, current_pos2, action2 = MoveAgent(board, Q_p2,epsilon,p2)
            r_p2 = 0
            gameOver, winner = CheckGameOver(board)
            if(gameOver):
                if(winner == p2):
                    r_p2 = 1
                elif(winner == p1):
                    r_p2 = -1
                Q_p2 = UpdateQ(board,prev_state,current_pos2,Q_p2,alpha,r_p2)
                break
            
            
            Q_p1 = UpdateQ(board,prev_state,current_pos1,Q_p1,alpha,r_p1)
            Q_p2 = UpdateQ(board,prev_state,current_pos2,Q_p2,alpha,r_p2)

            #################
            if(board.tobytes() in Q_p1 and board.tobytes() in Q_p2):
                print(f"board:\n {board}, \nq1:\n {Q_p1[board.tobytes()]}")
                print(f"board:\n {board}, \nq1:\n {Q_p2[board.tobytes()]}")
            
        #print(Q_p1)



if __name__ == "__main__":
    main()