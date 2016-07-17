import game
import numpy as np

### Test MoveScore ###

board = np.zeros([9])
move = np.zeros([9])
move[0] = 1
assert(game.MoveScore(board, move) == 0), game.MoveScore(board, move)

board = np.array([1,1,0,0,0,0,0,0,0])
assert(game.MoveScore(board, move) == -1), game.MoveScore(board, move)

board =  np.array([0,1,1,0,0,0,0,0,0])
assert(game.MoveScore(board, move) == 1), game.MoveScore(board, move)

board =  np.array([0,1,1,-1,-1,0, 0,0,0])
assert(game.MoveScore(board, move) == 1), game.MoveScore(board, move)

### Test IsValidMove ###
board = np.zeros([9])
move = np.array([1,0,0,0,0,0,0,0,0])
assert(game.IsValidMove(board, move)), game.IsValidMove(board, move)

board = [1,1,1,0,1,0,0,1,1]
assert(not game.IsValidMove(board, move)), game.IsValidMove(board, move)

board = [-1,1,0,-1,0,1,0,-1,1]
assert(len(game.PossibleValidMoves(board)) == 3), game.PossibleValidMoves(board)

print("Tests pass :)")
