import tensorflow as tf
import numpy as np
import random as rand

import game

NUM_ACTIONS = 9
GAMMA = 0.5

NUM_TRAINING_GAMES = 1000


def CreateNetwork():
    input_layer = tf.placeholder("float", [None, 9])
    layer_weights = tf.Variable(tf.zeros([9, 9]))
    bias = tf.Variable(tf.zeros([9]))
    output_layer = tf.nn.softmax(tf.matmul(input_layer, layer_weights) + bias)
    return input_layer, output_layer


def TrainModel():
# Number of games to play
    for i in range(NUM_TRAINING_GAMES):
        PlayGame()
        if i % 100 == 0:
            input("enter")

def PlayGame():
    board = np.zeros(9).tolist()

    states = []
    actions = []
    rewards = []

    while True:
        # look at possible moves
        for action in game.PossibleMoves(board):
            new_board_array = [[board[x] + action[x] for x in range(len(board))]]
            action_reward = session.run(output_layer, feed_dict={input_layer : new_board_array})[0]
            assert(type(board) == type([])) ,type(board)
            states.append(board)
            actions.append(action)
            rewards.append(game.MoveScore(board, action) + GAMMA * np.max(action_reward))
        result, board = UpdateBoard(board)

        if not result: break

    # train
    session.run(train_operation, feed_dict={input_layer: states, tf_actions: actions, targets: rewards})
    test_board = [np.zeros(9).tolist()]
    test_board[0][0] = 1
    test_board[0][1] = 1
    test_board[0][3] = -1
    test_board[0][4] = -1
    test_reward = session.run(output_layer, feed_dict={input_layer : test_board})[0]
    print(np.reshape(test_reward, (3,3)))

def UpdateBoard(board):
    valid_moves = game.PossibleValidMoves(board)
    if (len(valid_moves) == 0): return False, None
    chosen_move = rand.choice(valid_moves)
    if game.MoveScore(board, chosen_move) == 1: return False, None
    board = [board[x] + chosen_move[x] for x in range(len(board))]
    board = game.SwitchPlayer(board)
    return True, board

session = tf.Session()
input_layer, output_layer = CreateNetwork()

tf_actions = tf.placeholder("float", [None, NUM_ACTIONS])
targets = tf.placeholder("float", [None])
readout_action = tf.reduce_sum(tf.mul(output_layer, tf_actions), reduction_indices = 1)

cost = tf.reduce_mean(tf.square(targets - readout_action))
train_operation = tf.train.AdamOptimizer(0.01).minimize(cost)

session.run(tf.initialize_all_variables())
TrainModel()

state = [np.zeros(9).tolist()]
state[0][0] = 1
state[0][1] = 1

action_reward = session.run(output_layer, feed_dict={input_layer : state})
print(action_reward)
