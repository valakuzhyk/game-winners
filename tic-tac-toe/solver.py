import tensorflow as tf
import numpy as np
import random as rand
import matplotlib.pyplot as plt

import game

NUM_ACTIONS = 9
GAMMA = 0.7
LEARNING_RATE = 0.01

NUM_TRAINING_GAMES = 10000
TRAIN_STEP = 10
PRINT_STEP = 1000

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def CreateNetwork():
    input_layer = tf.placeholder("float", [None, 9])
    layer_weights = weight_variable([9,50])
    bias = bias_variable([50])
    
    hidden_layer = tf.nn.relu(tf.matmul(input_layer, layer_weights) + bias)

    layer2_weights = weight_variable([50,25])
    bias2 = bias_variable([25])
    hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, layer2_weights) + bias2)

    layer3_weights = weight_variable([25,18])
    bias3 = bias_variable([18])
    hidden_layer3 = tf.matmul(hidden_layer2, layer3_weights) + bias3

    layer4_weights = weight_variable([18,9])
    bias4 = bias_variable([9])

    output_layer = tf.nn.softmax(tf.matmul(hidden_layer3, layer4_weights) + bias4)
    return input_layer, output_layer

def CurrentState():
    # Check the results of training.
    test_board = [np.zeros(9).tolist()]
    test_board[0][0] = 1
    test_board[0][1] = 1
    test_board[0][2] = 0
    test_board[0][3] = -1
    test_board[0][4] = -1
    test_board[0][5] = 0
    test_board[0][6] = 1
    test_board[0][7] = -1
    test_board[0][8] = -1
    test_reward = session.run(output_layer_, feed_dict={input_layer_ : test_board})[0]
    return test_reward

def TrainModel():
    # this contains global data
    move_score = []

# Number of games to play
    states = []
    actions = []
    rewards = []
    for i in range(NUM_TRAINING_GAMES):
        PlayGame(states, actions, rewards)
        if i % TRAIN_STEP == 0:
            # train
            print(len(states))
            session.run(train_operation, feed_dict={input_layer_: states, tf_actions: actions, targets: rewards})
            states = []
            actions = []
            rewards = []
            move_score.append(CurrentState())
        if i % PRINT_STEP == 0:
            print(np.reshape(move_score[-1],(3,3)))
            plt.plot(move_score)
            plt.show()
            input("enter")

def PlayGame(states, actions, rewards):
    test_board = np.zeros(9).tolist()
    test_board[0] = -1
    test_board[1] = 0
    test_board[2] = 0
    test_board[3] = 1
    test_board[4] = 1
    test_board[5] = 0
    test_board[6] = -1
    test_board[7] = 1
    test_board[8] = 0
    board = test_board

    while True:
        # look at possible moves
        round_scores = []
        for action in game.PossibleMoves(board):
            new_board_array = [[board[x] + action[x] for x in range(len(board))]]
            action_reward = session.run(output_layer_, feed_dict={input_layer_ : new_board_array})[0]
            if not game.IsValidMove(board, action):
                reward_score = -10
            else:
               reward_score = game.MoveScore(board, action) + GAMMA * np.max(action_reward)

             
            print("board")
            print(np.reshape(board, (3,3)))
            print("action")
            print(np.reshape(action, (3,3)))
            print("reward")
            action_reward_score = np.max(action_reward)
            print(reward_score)
            print(action_reward_score)
            input("enter")
             

            states.append(board)
            actions.append(action)
            rewards.append(reward_score)
            round_scores.append(reward_score)
        result, board = UpdateBoard(board, round_scores.index(max(round_scores)))
        if not result: break

def UpdateBoard(board, max_score_move):
    assert(len(board) == 9), print(board)
    valid_moves = game.PossibleValidMoves(board)
    if (len(valid_moves) == 0): return False, None
    chosen_move = [0,0,0,0,0,0,0,0,0]
    chosen_move[max_score_move] = 1
    if game.MoveScore(board, chosen_move) == 1: return False, None
    board = [board[x] + chosen_move[x] for x in range(len(board))]
    board = game.SwitchPlayer(board)
    return True, board

session = tf.Session()
input_layer_, output_layer_ = CreateNetwork()

tf_actions = tf.placeholder("float", [None, NUM_ACTIONS])
targets = tf.placeholder("float", [None])
readout_action = tf.reduce_sum(tf.mul(output_layer_, tf_actions), reduction_indices = 1)

cost = tf.reduce_mean(tf.square(targets - readout_action))
#cost = tf.Print(cost, [cost], message="This is cost: ") 
train_operation = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

session.run(tf.initialize_all_variables())
TrainModel()
print("Done training")
