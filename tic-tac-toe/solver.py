import tensorflow as tf
import numpy as np
import random as rand

import game

NUM_ACTIONS = 9
GAMMA = 0.2

NUM_TRAINING_GAMES = 10000

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

    #drop_hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=0.9)

    layer2_weights = weight_variable([50,25])
    bias2 = bias_variable([25])
    hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, layer2_weights) + bias2)

    layer3_weights = weight_variable([25,18])
    bias3 = bias_variable([18])
    hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, layer3_weights) + bias3)

    layer4_weights = weight_variable([18,9])
    bias4 = bias_variable([9])

    output_layer = tf.nn.softmax(tf.matmul(hidden_layer3, layer4_weights) + bias4)
    return input_layer, output_layer

def CurrentState():
    # Check the results of training.
    test_board = [np.zeros(9).tolist()]
    test_board[0][0] = 1
    test_board[0][1] = 1
    test_board[0][7] = -1
    test_board[0][8] = -1
    test_reward = session.run(output_layer_, feed_dict={input_layer_ : test_board})[0]
    print(np.reshape(test_reward,(3,3)))
    input("enter")


def TrainModel():
# Number of games to play
    states = []
    actions = []
    rewards = []
    for i in range(NUM_TRAINING_GAMES):
        PlayGame(states, actions, rewards)
        if i % 1000 == 1:
            # train
            print(len(states))
            print(len(actions))
            print(len(rewards))
            session.run(train_operation, feed_dict={input_layer_: states, tf_actions: actions, targets: rewards})
            states = []
            actions = []
            rewards = []
            CurrentState()

def PlayGame(states, actions, rewards):
    board = np.zeros(9).tolist()
    board[0] = -1.;
    board[1] = 0.;
    board[2] = -1.;
    board[3] = 0.;
    board[4] = -1.;
    board[5] = 0.;
    board[6] = 1.;
    board[7] = 1;

    while True:
        # look at possible moves
        for action in game.PossibleMoves(board):
            new_board_array = [[board[x] + action[x] for x in range(len(board))]]
            action_reward = session.run(output_layer_, feed_dict={input_layer_ : new_board_array})[0]
            
            reward_score = game.MoveScore(board, action) + GAMMA * np.max(action_reward)
            if reward_score == 1:
                reward_score[-1] = -1

            '''
            print("board")
            print(np.reshape(board, (3,3)))
            print("action")
            print(np.reshape(action, (3,3)))
            print("reward")
            action_reward_score = np.max(action_reward)
            print(reward_score)
            print(action_reward_score)
            input("enter")
            '''

            states.append(board)
            actions.append(action)
            rewards.append(reward_score)
        result, board = UpdateBoard(board)
        if not result: break


def UpdateBoard(board):
    assert(len(board) == 9), print(board)
    valid_moves = game.PossibleValidMoves(board)
    if (len(valid_moves) == 0): return False, None
    chosen_move = rand.choice(valid_moves)
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
cost = tf.Print(cost, [cost], message="This is cost: ") 
train_operation = tf.train.AdamOptimizer(0.1).minimize(cost)

session.run(tf.initialize_all_variables())
TrainModel()
print("Done training")
