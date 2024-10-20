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
    return new_board,pos, action

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

def main():

    #### Set parameters ####
    p1 = 1 # 'X'
    p2 = -1 # 'O'

    Q_p1 = {}
    Q_p2 = {}
    epsilon = 1
    decay_rate = 0.9
    alpha = 0.1
    K =3 #10**4

    for k in range(0, K): 
        if(k % 100 == 0):
            epsilon *= decay_rate
        board = np.zeros((3,3))

        Q_p1, q1 = InitQ(Q_p1,board)
        Q_p2, q2 = InitQ(Q_p2,board)

        # both players put 1 piece, and they should have the same number of moves. thus range 0,4
        for t in range(0,5):
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
                action_new = BestMove(board,Q_p1) 

                bestpos = np.argwhere(q1 == action_new)
                if (isinstance(bestpos,np.ndarray)):
                    bestpos = ChooseEqualProb(bestpos)
                

                q1[current_pos1[0]][current_pos1[1]] += alpha*(r_p1+q1[bestpos[0]][bestpos[1]]-q1[current_pos1[0]][current_pos1[1]])
                Q_p1[board.tobytes()] = q1
                print(f"board:\n {board}, \nq1:\n {q1}")
                break
            
            else:
                action_new = BestMove(board,Q_p1) 

                bestpos = np.argwhere(q1 == action_new)
                if (isinstance(bestpos,np.ndarray)):
                    bestpos = ChooseEqualProb(bestpos)

                q1[current_pos1[0]][current_pos1[1]] += alpha*(r_p1+q1[bestpos[0]][bestpos[1]]-q1[current_pos1[0]][current_pos1[1]])
                Q_p1[board.tobytes()] = q1


            ## Player 2 ##
            board, current_pos2, action1 = MoveAgent(board, Q_p2,epsilon,p2)
            r_p2 = 0
            gameOver, winner = CheckGameOver(board)
            if(gameOver):
                if(winner == p2):
                    r_p2 = 1
                elif(winner == p1):
                    r_p2 = -1
            
                
            #################
            print(f"board:\n {board}, \nq1:\n {q1}")
            
        #print(Q_p1)



if __name__ == "__main__":
    main()