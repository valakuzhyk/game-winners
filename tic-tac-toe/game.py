# This is the first test of how to write an AI for playing games.
# This will be a first test of how to implement tensorflow algorithms as well
# as using git for source control.  Some of us are noobs here. 
# also, assume board is 3x3. Don't have patience for larger than that.
# also, don't care about invalid game scenarios (aka two people won)

# -----------------------------------------------------------------------------
# Check if a move is a win. 
def IsWon(board):
    assert(len(board) != 0)
    assert(len(board) == len(board[0]))
    # check horizontal
    for row in board:
        row_sum = sum(row)
        if row_sum == 3: return 1
        if row_sum == -3: return -1
    # check vertical
    for row_idx in range(len(board)):
        col_sum = 0
        for col_idx in range(len(board)):
            col_sum += board[row_idx][col_idx]
        if row_sum == 3: return 1
        if row_sum == -3: return -1
    # check diagonal
    diag_sum1 = 0
    diag_sum2 = 0
    for i in range(len(board)):
            diag_sum1 += board[i][i]
            diag_sum2 += board[i][len(board)-i-1]
    if (diag_sum1 == 3 or diag_sum2 == 3): return 1
    if (diag_sum1 == -3 or diag_sum2 == -3): return -1
    return 0

# -----------------------------------------------------------------------------
# Returns list of possible move locations
def FindPossibleMoves(board):
    moves = []
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == 0:
                moves.append([r,c])
    return moves

# -----------------------------------------------------------------------------

