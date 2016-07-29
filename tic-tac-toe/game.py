import numpy as np

# This is the first test of how to write an AI for playing games.
# This will be a first test of how to implement tensorflow algorithms as well
# as using git for source control.  Some of us are noobs here. 

# -----------------------------------------------------------------------------
# Give the score for a move. Invalid moves will be given a score of -inf.
def MoveScore(board_before_move, move):
    assert(len(board_before_move) == 9)
    assert(len(move) == 9)
    if not IsValidMove(board_before_move, move):
        return -10
    board = np.array(board_before_move) + np.array(move)

    # check horizontal
    if sum(board[0:3]) > 2.9 or sum(board[3:6]) > 2.9 or sum(board[6:9]) > 2.9:
        return 1

    # check vertical
    for i in range(3):
        col_sum = board[i] + board[i+3] + board[i+6]
        if col_sum > 2.9: return 1

    # check diagonal
    diag_sum1 = board[0] + board[4] + board[8]
    diag_sum2 = board[2] + board[4] + board[6]
    if (diag_sum1 > 2.9 or diag_sum2 > 2.9): return 1
    return 0

# -----------------------------------------------------------------------------
# Returns true if this move can be made
def IsValidMove(board, move):
    assert(len(board) == 9)
    assert(len(move) == 9)
    for idx, val in enumerate(move):
        if (val == 1 and (board[idx] == 1 or board[idx] == -1)):
            return False
    return True

# -----------------------------------------------------------------------------
# Switches the board for the next player.
def SwitchPlayer(board):
    assert(len(board) == 9)
    return [-x for x in board]

def PossibleMoves(board):
    assert(len(board) == 9)
    moves = []
    for i in range(len(board)):
        move = np.zeros(9).tolist()
        move[i] = 1
        moves.append(move)
    return moves

def PossibleValidMoves(board):
    assert(len(board) == 9), print(board)
    moves = PossibleMoves(board)
    valid_moves = []
    for move in moves:
        if IsValidMove(board, move):
            valid_moves.append(move)
    return valid_moves
