"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # First flatten the list of lists, then count and compare the numbers of X and O
    if sum(board,[]).count(X) == sum(board,[]).count(O):
        return X
    else: 
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    #Checks for empty spots on the board and assigns each empty spot a coordinate
    moves = set()
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == EMPTY:
                moves.add((i,j))
           
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    #Copy the original board and create dummy indices for the coordinates of the moves
    dummy = copy.deepcopy(board)
    i = action[0]
    j = action[1]

    #Check for moves that are outside of the board or would act onto a non-empty spot
    if dummy[i][j] != EMPTY:
        raise Exception("This move is not allowed.")
    elif i not in range(len(board)) or j not in range(len(board)):
        raise Exception("This move is not within the board.")
    else:
        dummy[i][j] = player(board)
    return dummy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    #Transpose the board to exchange rows and colums
    tboard = list(zip(*board))
    #Checks the rows
    for i in range(len(board)):
        if board[i].count(board[i][0]) == len(board[i]):
            return board[i][0]
        elif tboard[i].count(tboard[i][0]) == len(tboard[i]):
            return tboard[i][0]
    #Check the diagonals
    dummy1 = []
    dummy2 = []

    for i in range(len(board)):
        dummy1.append(board[i][i])
        dummy2.append(board[i][len(board)-1-i])

    if dummy1.count(dummy1[0]) == len(dummy1):
        return dummy1[0]
    elif dummy2.count(dummy2[0]) == len(dummy2):
        return dummy2[0]

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None or sum(board,[]).count(EMPTY) == 0:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    def max_value(board):
        #If the game is over, return score and no move
        if terminal(board):
            return utility(board), None
        
        v = -42
        move = None

        for action in actions(board):
            V, Move = min_value(result(board,action))
            if V > v:
                v = V
                move = action
                if v == 1:
                    return v, move

        return v, move

    def min_value(board):
        #If the game is over, return score and no move
        if terminal(board):
            return utility(board), None
        
        v = 42
        move = None

        for action in actions(board):
            V, Move = max_value(result(board,action))
            if V < v:
                v = V
                move = action
                if v == -1:
                    return v, move

        return v, move
              
    if terminal(board):
        return None
    else:
        if player(board) == X:
            return max_value(board)[1]
        else:
            return min_value(board)[1]