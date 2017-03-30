import snake
import numpy as np
import random
import tensorflow as tf
from  collections import deque
import os
import sys
 

MAX_GAME = 100000
LEARN_RATE = 1e-6
MINI_BATCH_SIZE = 100
FUTURE_REWARD_DISCOUNT = 0.95
MEMORY_SIZE = 500000
OBSERVATION_STEPS = 1000

GAME_WIDTH = 14
GAME_HEIGHT = 14
FEATURE_NUMBER = 3


INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
FINAL_RANDOM_ACTION_PROB = 0.001  # final chance of an action being random
 
probability_of_random_action = INITIAL_RANDOM_ACTION_PROB


def conv(x, k, out_dim, name):
    with tf.name_scope(name):
        kernel_shape = [k, k, int(x.get_shape()[-1]), out_dim]
        with tf.name_scope('weights'):
            w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01))
        with tf.name_scope('biases'):
            b = tf.Variable(tf.constant(0.01, shape=[out_dim]))
        with tf.name_scope('conv2d'):
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
            out = tf.nn.bias_add(conv, b)
        with tf.name_scope('relu'):
            result = tf.nn.relu(out)
            return result

def maxpool(x, k, name):
    with tf.name_scope('max_pool_%s' % name):
        kernel_shape = [1, k, k, 1]
        return tf.nn.max_pool(x, kernel_shape, strides=[1,1,1,1], padding='SAME', name='max_pool')

def linear_layer(x, out_size, name):
    with tf.name_scope('linear_%s' % name):
        shape = x.get_shape().as_list()
        weights = tf.Variable(tf.truncated_normal([shape[1], out_size], stddev=0.1))
        bias = tf.Variable(tf.constant(0.0, shape=[out_size]))
        out = tf.nn.bias_add(tf.matmul(x, weights, name='matmul'), bias, name='bias_add')
        return out 

def create_network(W, H, C):
    with tf.name_scope('step'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        step_input = tf.placeholder('int32', None, name='step_input')
        step_assign_op = global_step.assign(step_input)
    
    action_size = 4
    input = tf.placeholder("float", [None, C, H, W], name="input")
    input_tr = tf.transpose(input, [0,2,3,1])
    conv1 = conv(input_tr, 5, 32, 'c1')
    pool1 = maxpool(conv1, 2, 'c1_pool')
    conv2 = conv(pool1, 3, 64, 'c2')

    shape = conv2.get_shape().as_list()
    conv2_flat = tf.reshape(conv2, [-1, reduce(lambda x, y: x * y, shape[1:])])

    l1=  tf.nn.relu(linear_layer(conv2_flat, 512, 'hidden'))
    q =  linear_layer(l1, action_size, 'output') # 4 is action size
    

    ##########################################################################
    target = tf.placeholder("float", [None], name = "target")
    action = tf.placeholder('int64', [None], name = "action")

    with tf.name_scope('delta'):
        action_one_hot = tf.one_hot(action, action_size, 1.0, 0.0)
        masked_actions = tf.reduce_sum(q * action_one_hot, reduction_indices=1)
        delta = target - masked_actions
        clipped_delta = tf.clip_by_value(delta, -1, 1) 

    with tf.name_scope('square_loss'):
        loss = tf.reduce_mean(tf.square(clipped_delta))
    
    with tf.name_scope('train'): 
        optim = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)

    summary_placeholders = {}
    summary_ops = {}
    for tag in ['average.loss', 'average.q', 'average.reward', 'average.move']:
        summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
        summary_ops[tag] = tf.scalar_summary(tag, summary_placeholders[tag])

    params = {
        'input':input,
        'output':q,
        'target':target,
        'action':action,
        'optim':optim,
        'loss':loss,
        'summary_placeholders':summary_placeholders,
        'summary_ops':summary_ops,
        'step':global_step,
        'step_input':step_input,
        'step_op':step_assign_op,
    }
    return params

def build_state(game):
    output = np.zeros((FEATURE_NUMBER, game.height, game.width))
    for y in range(game.height):
        for x in range(game.width):
            offset = y * game.width + x
            output[2][y][x] = game.player_board[offset]
            output[1][y][x] = game.object_board[offset]
    
    head = game.player_chain[0]
    h_y = int(head / game.width)
    h_x = head % game.width 
    output[0][h_y][h_x] = 1

    return output

def get_next_action(s_t, probability_of_random=0):
    # random action
    if random.random() > probability_of_random:
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
            rewards_expected.append(max(0, np.max(rewards_next[i])) * FUTURE_REWARD_DISCOUNT + rewards[i]) 

    _, loss, output = sess.run([params['optim'], params['loss'], params['output']], feed_dict={params['input']:states, params['action']:actions, params['target']:rewards_expected})
    return loss, output.mean()

def write_loss_q_summary(params, step, avg_loss, avg_q):
    loss_ops = params['summary_ops']['average.loss']
    q_ops = params['summary_ops']['average.q']
    loss_placeholder = params['summary_placeholders']['average.loss']
    q_placeholder = params['summary_placeholders']['average.q']

    summary_str_lists = sess.run([loss_ops, q_ops], feed_dict={loss_placeholder:avg_loss,q_placeholder:avg_q}) 
    for summary_str in summary_str_lists:
        writer.add_summary(summary_str, step)

def write_reward_move_summary(params, step, reward, move):
    reward_ops = params['summary_ops']['average.reward']
    move_ops = params['summary_ops']['average.move']
    reward_placeholder = params['summary_placeholders']['average.reward']
    move_placeholder = params['summary_placeholders']['average.move']

    summary_str_lists = sess.run([reward_ops, move_ops], feed_dict={reward_placeholder:reward,move_placeholder:move}) 
    for summary_str in summary_str_lists:
        writer.add_summary(summary_str, step)


def main(game, screen, memory, params):
    global probability_of_random_action
    score = 0
    last_state = build_state(game)
    loop = 0
    total_loss = 0
    total_q = 0
    while True:
        game.draw(screen)
        # check player input
        action = get_next_action(last_state, probability_of_random_action)
        game.player_direction = action
        # update
        reward = game.update()
        # get post state
        state = build_state(game)
        terminal = reward != 0
        
        memory.append((last_state, action, max(0, reward), state, terminal))
         
        last_state = state
        if reward <= -1:
            break
        else:
            score += reward
            if loop > 500: 
                break

        if len(memory) > OBSERVATION_STEPS:
            loss, q_t = train(memory, params)
            total_loss += loss
            total_q = q_t
            probability_of_random_action = max(probability_of_random_action - 0.00001, FINAL_RANDOM_ACTION_PROB)
            
            step = params['step'].eval() + 1
            params['step_op'].eval({params['step_input']: step})

            if step % 100 == 0:
                write_loss_q_summary(params, step, total_loss/100, total_q/100)
                total_loss = 0
                total_q = 0

        while len(memory) > MEMORY_SIZE:
            memory.popleft()
        
        loop += 1
    return score, loop

if __name__ == '__main__':
    model_file = 'snake-model9'
    screen = None#snake.init()
    memory = deque()
    sess = tf.Session()
    with sess.as_default():

        params = create_network(GAME_HEIGHT, GAME_WIDTH, FEATURE_NUMBER)

        total_score = 0.0
        total_loop = 0.0
        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter('./summary/%s' % model_file, tf.get_default_graph())
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
            write_reward_move_summary(params, ngame, score, loop)
            
            if ngame % 100 == 0:
                print('avg score is %f avg game loop %f, prob %f' % (total_score*0.01, total_loop*0.01, probability_of_random_action))
                sys.stdout.flush()
                total_score = 0
                total_loop = 0

            if ngame % 1000 == 0:
                step = params['step'].eval()
                saver.save(sess, './save/' + model_file,global_step=step)
    
