import numpy as np

# This is the first test of how to write an AI for playing games.
# This will be a first test of how to implement tensorflow algorithms as well
# as using git for source control.  Some of us are noobs here. 
# also, assume board is 3x3. Don't have patience for larger than that.
# also, don't care about invalid game scenarios (aka two people won)

def ArrayToMatrix(board):
    return  np.reshape(board, (3,3))

def MatrixToArray(board):
    return  np.reshape(board, 9)

# -----------------------------------------------------------------------------
# Give the score for a move. Invalid moves will be given a score of -inf.
def MoveScore(matrix_board, matrix_move):
    board_before_move = ArrayToMatrix(matrix_board)
    move = ArrayToMatrix(matrix_move)
    assert(len(board_before_move) != 0)
    if not IsValidMove(board_before_move, move):
        return -10.
    board = board_before_move + move

    # check horizontal
    for row in board:
        row_sum = sum(row)
        if row_sum == 3: return 1
    # check vertical
    for row_idx in range(len(board)):
        col_sum = 0
        for col_idx in range(len(board)):
            col_sum += board[row_idx][col_idx]
        if row_sum == 3: return 1
    # check diagonal
    diag_sum1 = 0
    diag_sum2 = 0
    for i in range(len(board)):
            diag_sum1 += board[i][i]
            diag_sum2 += board[i][len(board)-i-1]
    if (diag_sum1 == 3 or diag_sum2 == 3): return 1
    return 0

# -----------------------------------------------------------------------------
# Returns true if this move can be made
def IsValidMove(board, move):
    array_board = MatrixToArray(board)
    array_move = MatrixToArray(move)
    if sum(array_board * array_move) != 0:
        return False
    return True

# -----------------------------------------------------------------------------
# Switches the board for the next player.
def SwitchPlayer(board):
    return [-x for x in board]

def PossibleMoves(board):
    moves = []
    for i in range(len(board)):
        move = np.zeros(9).tolist()
        move[i] = 1
        moves.append(move)
    return moves

def PossibleValidMoves(board):
    valid_moves = []
    moves = PossibleMoves(board)
    for move in moves:
        if IsValidMove(board, move):
            valid_moves.append(move)
    return valid_moves
