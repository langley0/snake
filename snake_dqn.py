import snake
import numpy as np
import random
import tensorflow as tf
from  collections import deque
import os 

MAX_GAME = 100000
LEARN_RATE = 1e-6
MINI_BATCH_SIZE = 100
FUTURE_REWARD_DISCOUNT = 0.95
MEMORY_SIZE = 500000
OBSERVATION_STEPS = 1000

GAME_WIDTH = 10
GAME_HEIGHT = 10


INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
FINAL_RANDOM_ACTION_PROB = 0.001  # final chance of an action being random
 
probability_of_random_action = INITIAL_RANDOM_ACTION_PROB

def conv(x, k, out_dim):
    kernel_shape = [k, k, int(x.get_shape()[-1]), out_dim]
    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01))
    b = tf.Variable(tf.constant(0.01, shape=[out_dim]))
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    out = tf.nn.bias_add(conv, b)
    return tf.nn.relu(out)

def maxpool(x, k):
    kernel_shape = [1, k, k, 1]
    return tf.nn.max_pool(x, kernel_shape, strides=[1,1,1,1], padding='SAME')


def linear_layer(x, out_size, name=None):
    shape = x.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([shape[1], out_size], stddev=0.1))
    bias = tf.Variable(tf.constant(0.0, shape=[out_size]))
    out = tf.nn.bias_add(tf.matmul(x, weights), bias, name=name)
    return out 

def create_network(W, H, C):
    action_size = 4
    input = tf.placeholder("float", [None, C, H, W], name="input")
    input_tr = tf.transpose(input, [0,2,3,1])
    conv1 = conv(input_tr, 5, 32)
    pool1 = maxpool(conv1, 2)
    conv2 = conv(pool1, 3, 64)

    shape = conv2.get_shape().as_list()
    conv2_flat = tf.reshape(conv2, [-1, reduce(lambda x, y: x * y, shape[1:])])

    l1=  tf.nn.relu(linear_layer(conv2_flat, 512))
    q =  linear_layer(l1, action_size) # 4 is action size 

    ##########################################################################
    target = tf.placeholder("float", [None], name = "target")
    action = tf.placeholder('int64', [None], name = "action")

    action_one_hot = tf.one_hot(action, action_size, 1.0, 0.0)
    masked_actions = tf.reduce_sum(q * action_one_hot, reduction_indices=1)
    delta = target - masked_actions
    clipped_delta = tf.clip_by_value(delta, -1, 1) 

    loss = tf.reduce_mean(tf.square(clipped_delta), name='loss')
    optim = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss, name ="train_op")

    return input, q, target, action, optim

def build_state(game):
    output = np.zeros((game.height, game.width))
    for y in range(game.height):
        for x in range(game.width):
            offset = y * game.width + x
            if (game.player_board[offset] == 1):
                output[y][x] = 1
            if (game.object_board[offset] == 1):
                output[y][x] = 1
    return output

def get_next_action(s_t, probability_of_random=0):
    # random action
    if s_t != None and random.random() > probability_of_random:
        output = sess.run(params['output'], feed_dict={params['input']:[s_t]})[0]
        return np.argmax(output)
    else:
        valid_actions = [snake.ACTION_LEFT, snake.ACTION_RIGHT, snake.ACTION_UP, snake.ACTION_DOWN]
        return valid_actions[random.randint(0, len(valid_actions)-1)]

def train(memory, params):
    mini_batch = random.sample(memory, MINI_BATCH_SIZE)
    states = [d[0] for d in mini_batch]
    actions = [d[1] for d in mini_batch]
    rewards = [d[2] for d in mini_batch]
    states_next = [d[3] for d in mini_batch]
    terminals = [d[4] for d in mini_batch]
     
    rewards_expected = []
    rewards_next = sess.run(params['output'], feed_dict={params['input']:states_next})
    for i in range(MINI_BATCH_SIZE):
        terminal = terminals[i]
        if terminal:
            rewards_expected.append(rewards[i])
        else:
            rewards_expected.append(np.max(rewards_next[i]) * FUTURE_REWARD_DISCOUNT + rewards[i]) 

    sess.run(params['optim'], feed_dict={params['input']:states, params['action']:actions, params['target']:rewards_expected})

def main(game, screen, memory, params):
    global probability_of_random_action
    score = 0
    history = None
    loop = 0
    _action_history = []
    while True:
        game.draw(screen)
        # check player input
        action = get_next_action(history, probability_of_random_action)
        game.player_direction = action
        _action_history.append(action)
        player_prev = game.player_chain[0]
        object_prev = game.object_pos
        # update
        reward = game.update()
        # get post state
        state = build_state(game)
        player_post = game.player_chain[0]
        object_post = game.object_pos

        if reward < 0:
            terminal = True
        else:
            terminal = False
        
        if history != None:
            s_t = history[:]
            history = history[1:] + [state]
            s_t_1 = history[:]

            memory.append((s_t, action, reward, s_t_1, terminal))
        else:
            history = [state] * 4
 
        score += reward 
        if reward < 0:
            return score, loop

        if len(memory) > OBSERVATION_STEPS:
            train(memory, params)
            probability_of_random_action = max(probability_of_random_action - 0.00001, FINAL_RANDOM_ACTION_PROB)

        while len(memory) > MEMORY_SIZE:
            deque.popleft()
        
        loop += 1

if __name__ == '__main__':
    screen = None#snake.init()
    memory = deque()
    sess = tf.Session()
    with sess.as_default():

        input, output, target, action, optim = create_network(GAME_HEIGHT, GAME_WIDTH, 4) 
        params = {
            'input':input,
            'output':output,
            'target':target,
            'action':action,
            'optim':optim
        }

        total_score = 0.0
        total_loop = 0.0
        saver = tf.train.Saver()
        
        model_file = 'snake-model'
        if os.path.exists(model_file):
            print('model restored')
            saver.restore(sess, model_file)
        else:
            print('model initialized')
            sess.run(tf.initialize_all_variables())
        for ngame in range(MAX_GAME): 
            game = snake.SnakeGame(width=GAME_WIDTH, height=GAME_HEIGHT)
            score, loop = main(game, screen, memory, params)
            total_score += score
            total_loop += loop 
            if ngame % 100 == 0:
                print('avg score is %d avg game loop %d, prob %f' % (total_score, total_loop, probability_of_random_action))
                total_score = 0
                total_loop = 0

                saver.save(sess, model_file)
    