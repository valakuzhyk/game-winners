import game
import numpy

### Test IsWon

board = numpy.zeros([3,3])
assert(game.IsWon(board) == 0)

board = [[1,1,1],[0,0,0],[0,0,0]]
assert(game.IsWon(board) == 1)

board = [[-1,0,0],[0,-1,0],[0,0,-1]]
assert(game.IsWon(board) == -1)

### TestFindPossibleMoves
board = numpy.zeros([3,3])
assert(len(game.FindPossibleMoves(board)) == 9)

board = [[1,1,1],[0,1,0],[0,1,1]]
moves = game.FindPossibleMoves(board)
assert(len(moves) == 3)
assert([1,0] in moves)
assert([1,2] in moves)
assert([2,0] in moves)

print("Tests pass :)")
