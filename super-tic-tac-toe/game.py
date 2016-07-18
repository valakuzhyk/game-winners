import numpy as np

# -----------------------------------------------------------------------------
def IsBoardWon(board):
    # Rows
    for row in board:
        if row.count(1) == 3: return 1
        if row.count(-1) == 3: return -1
    # Columns
    for i in range(3):
        if board[i:9:3].count(1) == 3: return 1
        if board[i:9:3].count(-1) == 3: return -1
    # Diags
    if board[0:9:4].count(1) == 3 or board[2:8:2].count(1) == 3: return 1
    if board[0:9:4].count(-1) == 3 or board[2:8:2].count(-1) == 3: return -1

    return 0

# -----------------------------------------------------------------------------
def IsBoardDone(board):
    return IsBoardWon(board) or board.count(0)

# -----------------------------------------------------------------------------
def GetMetaBoard(board):
    return [IsBoardWon(board[i*9: i*9 + 9]) for i in range(9)]

# -----------------------------------------------------------------------------
# Returns 1 if the move wins, 0 if it doesn't, -2 if it is illegal.
def MoveScore(board_before_move, move):
    assert(move.count(0) == 2)
    if not IsValidMove(board_before_move, move):
        return -2

    playing_board = GetBoardN(board_before_move, move[:9].index(1))
    board = board_before_move[:]
    board[move[9:].index(1) * 9 + move[9:].index(1)] = 1

    meta_board = GetMetaBoard(board)
    if (IsBoardWon(meta_board)): return 1
    return 0

# -----------------------------------------------------------------------------
def IsValidMove(board_set, move):
    # First, check whether the move is playing on a valid board
    valid_boards = board_set[81:]
    chosen_board = move[9:]
    if valid_boards[chosen_board.index(1)] != 1:
        return False

    # second, check whether the move is played in a valid square in that board
    board = board_set[chosen_board.index(1)*9 : chosen_board.index(1) * 9 + 9]
    board_move = move[:9]
    if (board[board_move.index(1)] != 0):
        return False

    return True

# -----------------------------------------------------------------------------
def SwitchPlayer(board):
    assert(type(board) == type([]))
    new_board = []
    for val in board[0:81]:
        new_board.append(-val)
    new_board += board[81:]
    return new_board

# -----------------------------------------------------------------------------
# Create all possible moves
def PossibleMoves():
    moves = []
    for i in range(9):
        for j in range(9):
            move = np.zeros((1,18)).tolist()
            move[i] = 1
            move[9+j] = 1
            moves.append(move)
    return moves 

    
# -----------------------------------------------------------------------------
# Choose only the possible moves that are valid.
def PossibleValidMoves(board):
    return [x for x in PossibleMoves(board) if IsValidMove(board, x)]

# -----------------------------------------------------------------------------
def CreateBoard():
    # The first 9 values indicate which boards are valid to play on.
    # If the corresponding value is 1, you may play on the board.
    board = np.zeros(90)
    board[81:] = 1
    return board.tolist()

# -----------------------------------------------------------------------------
def UpdateBoard(board, move):
    assert(IsValidMove(board, move))
    assert(move.count(1) == 2)
    chosen_board = move[9:].index(1)

    move_idx = move[:9].index(1)

    board[chosen_board*9 + move_idx] = 1

    # Check whether the indicated board is playable
    next_board = GetBoardN(board, move_idx)
    if IsBoardDone(next_board):
        board[81:] = [1] * 9
    else:
        board[81:] = move_idx
    
def GetBoardN(board, idx):
    return board[idx*9: idx*9 + 9]


    
