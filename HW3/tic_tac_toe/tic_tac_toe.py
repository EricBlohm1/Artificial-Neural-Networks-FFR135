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

        

def main():

    #### Set parameters ####
    p1 = 1 # 'X'
    p2 = -1 # 'O'

    Q_p1 = {}
    Q_p2 = {}
    epsilon = 1
    decay_rate = 0.9
    alpha = 0.1
    K = 1 #10**4

    for k in range(0, K): 
        if(k % 100 == 0):
            epsilon *= decay_rate
        board = np.zeros((3,3))


        if(len(Q_p1) == 0):
            q1 = np.zeros((3,3))
            Q_p1[board.tobytes()] = q1
        else:
            q1= Q_p1[board.tobytes()].copy()


        if(len(Q_p2) == 0):
            q2 = np.zeros((3,3))
            Q_p2[board.tobytes()] = q2
        else:
            q2= Q_p2[board.tobytes()].copy()

        for t in range(0,9):
            ### Player 1 ###
            new_board, current_pos, action = MoveAgent(board, Q_p1,epsilon,p1) ## i have no clue why. Returns shape (array([ ...]), None) ??

            print(f"new board:\n {new_board}, \nq1:\n {q1}")
            r_p1 = 0
            gameOver, winner = CheckGameOver(new_board)
            if(gameOver):
                if(winner == 1):
                    r_p1 = 1
                elif(winner == -1):
                    r_p1 = -1
                #new position
                action_new = BestMove(new_board,Q_p1) # can return multiple of the same
                bestpos = np.argwhere(q1 == action_new)[0]
                print("currentpos", current_pos)
                q1[current_pos[0]][current_pos[1]] += alpha*(r_p1+q1[bestpos[0]][bestpos[1]]-q1[current_pos[0]][current_pos[1]])

                break
            #################
            action_new = BestMove(new_board,Q_p1) # can return multiple of the same
            bestpos = np.argwhere(q1 == action_new)[0]
            q1[current_pos[0]][current_pos[1]] += alpha*(r_p1+q1[bestpos[0]][bestpos[1]]-q1[current_pos[0]][current_pos[1]])
            board = new_board
            Q_p1[board.tobytes()] = q1
        #print(Q_p1)



if __name__ == "__main__":
    main()