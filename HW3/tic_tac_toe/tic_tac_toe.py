import numpy as np



def update_board(board, prev, new, player):
    board[new[0]][new[1]] = player
    board[prev[0]][prev[1]] = 0
    #return board
    

def display_board(board):
    display_board = np.full((len(board),len(board[0])),'')
    for i in range(0,len(board)):
        for j in range(0,len(board[0])):
            if(board[i][j] == 1) :
                display_board[i][j] = 'X'
            elif (board[i][j] == -1):
                display_board[i][j] = 'O'
            else: 
                display_board[i][j] = '-'

    print('--------------')
    for row in display_board:
        print('  |  '.join(row))
        print('--------------')


def main():

    p1 = 1 # 'X'
    p2 = -1 # 'O'

    #Q_p1 = 
    #Q_p2 = 
    board = np.zeros((3,3))

    update_board(board,[1,1],[2,2],p1)

    update_board(board,[2,2],[0,0],p1)

    update_board(board,[2,2],[1,1],p2)
    print(board)

    display_board(board)

    test = np.nan
    print(np.isnan(test))




    print("")

if __name__ == "__main__":
    main()