from policygrad import PolicyGrad
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import os
import random

class LSTMPolicyGrad(PolicyGrad):

    lr = 0.002
    lr_decay = 0.999
    gamma = 0.95
    quotes = []
    positions= []
    rewards = []
    actions = []
    loss_func = None
    def __init__(self, input_sz_, positions, output_sz_ , chunk_size_, max_len_sent, rnn_size_,channels_, save_path=None, load_path=None):
        self.n_classes = output_sz_
        self.time_len = max_len_sent
        self.chunk_size = int(chunk_size_)
        self.n_chunks = int(max_len_sent / chunk_size_)
        self.rnn_size = rnn_size_
        # self.X = tf.placeholder('float', [None, self.n_chunks, self.chunk_size])
        self.nr_positions = positions
        self.channels = channels_
        self.X = tf.placeholder('float', [None,positions, max_len_sent,self.channels])
        self.Y = tf.placeholder('float', [None, self.nr_positions])

        self.pos = tf.placeholder('float', [None, self.nr_positions])
        self.discounted_rewards = tf.placeholder(tf.float32, [None, ], name="actions_value")
        self.input_sz = input_sz_
        self.output_sz = output_sz_

        logits = self.model_cnn(self.X,self.pos)
        # logits = self.simple_model(self.X)
        self.loss_function(logits)

        self.tf_sess.run(tf.global_variables_initializer())
        # self.save_path = None
        #
        # self.saver = tf.train.Saver()

        # if save_path is not None and os.path.isfile(os.path.abspath(save_path + '.meta')):
        #
        #     self.load_path = save_path
        #     try:
        #         self.saver.restore(self.tf_sess, self.load_path)
        #         print('model loaded')
        #     except:
        #         print('could not reload weights')
        #
        # if save_path is not None:
        #     self.save_path = save_path

    def model_cnn(self, x, pos):

        filter1 = tf.Variable(tf.random_normal([1,3,self.channels,4]))
        filter2 = tf.Variable(tf.random_normal([1, 6,4,16]))
        filter3 = tf.Variable(tf.random_normal([1, 6, 16, 32]))
        filter4 = tf.Variable(tf.random_normal([1, 6,32, 64]))
        filter5 = tf.Variable(tf.random_normal([1, self.time_len - 17, 64, 64]))
        # x = tf.Print(x, [x], "x")
        # pos = tf.Print(pos, [pos], "pos")
        # filter1 = tf.Print(filter1, [filter1], "layer_1")
        # filter2 = tf.Print(filter2, [filter2], "filter2")
        layer_1 = tf.nn.conv2d(x,filter1,[1,1,1,1],'VALID')
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.nn.conv2d(layer_1, filter2, [1, 1, 1, 1], 'VALID')
        layer_2 = tf.nn.relu(layer_2)
        layer_3 = tf.nn.conv2d(layer_2, filter3, [1, 1, 1, 1], 'VALID')
        layer_3 = tf.nn.tanh(layer_3)
        layer_4 = tf.nn.conv2d(layer_3, filter4, [1, 1, 1, 1], 'VALID')
        layer_4 = tf.nn.tanh(layer_4)
        layer_5 = tf.nn.conv2d(layer_4, filter5, [1, 1, 1, 1], 'VALID')
        layer_5 = tf.nn.tanh(layer_5)
        flat_tensor = tf.reshape(layer_5, [-1,self.nr_positions*64])

        combined = tf.concat((flat_tensor, pos),1)
        # print('combined sh ',combined.shape)
        # combined = tf.Print(combined, [combined], 'combined')
        # output = tf.contrib.layers.fully_connected(flat_tensor,  self.output_sz)
        # output = output/tf.reduce_sum(output)
        output = tf.layers.dense(inputs=flat_tensor,
                                 units=50,
                                 activation=None)
        output = tf.layers.dense(inputs=flat_tensor,
                                 units=self.output_sz,
                                 activation=None)
        # output = tf.Print(output, [output], "output")
        return output


    def simple_model(self, x):
        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                 'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.transpose(x, [1, 0, 2,1])
        x = tf.reshape(x, [-1, self.chunk_size])
        x = tf.split(x,self. n_chunks, 0)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

        return output
    # def model(self, x, pos):
    #
    #     layer_pos = {'weights': tf.Variable(tf.random_normal([self.nr_positions, 5])),
    #                  'biases': tf.Variable(tf.random_normal([5]))}
    #
    #     layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size+5, self.n_classes])),
    #              'biases': tf.Variable(tf.random_normal([self.n_classes]))}
    #
    #     x = tf.transpose(x, [1, 0, 2])
    #     x = tf.reshape(x, [-1, self.chunk_size])
    #     x = tf.split(x,self. n_chunks, 0)
    #
    #     lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
    #     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #
    #     pos_output = tf.matmul(pos, layer_pos['weights']) + layer_pos['biases']
    #     combined = tf.concat((outputs[-1],pos_output),1)
    #     output = tf.matmul(combined, layer['weights']) + layer['biases']
    #
    #     return output

    def gather_data(self, state, reward):

        self.quotes.append(state['quote'])
        self.rewards.append(reward)

    def normalize(self, rewards):
        disc_reward = np.zeros_like(rewards)
        cummulative = 0

        for t in reversed(range(len(rewards))):
            cummulative = cummulative * self.gamma + rewards[t]
            disc_reward[t] = cummulative

        disc_reward = disc_reward - np.mean(disc_reward)
        disc_reward = disc_reward / np.std(disc_reward)
        # print('disc_reward ',disc_reward)

        return disc_reward


    def get_action(self, observation):

        quote = observation['quote']
        position = observation['position']
        self.positions.append(np.array(position,dtype = np.float))

        quote = quote.reshape(-1,self.nr_positions, self.time_len,self.channels)
        position = np.array(position).reshape(-1, self.nr_positions)
        prob_weights =self.tf_sess.run(self.outputs_softmax, feed_dict={self.X: quote, self.pos: position})
        # print('prob_weights ',prob_weights, ' quote ',quote[0][0][-3:-1])
        # prob_weights = [[1, 1, 1, 1]]
        # prob_weights =[np.random.randint(4, size=1)]
        # # action=prob_weights
        # action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        # new_action = np.zeros(self.output_sz)
        # new_action[np.random.randint(3, size=1)] = 1
        # action= new_action
        sum_prob = np.sum(prob_weights[0])
        sum_prob = 1
        if sum_prob < 0.000001:
            action = [0.2,0.4,0.1,0.30]
            print('ZERO SUM PROBABILITY DETECTED')
        else:
            action = prob_weights[0] /sum_prob #make sure it sums to 1
        # # print('prob_weights',prob_weights,'get_action',action)
        # action = prob_weights[0]
        # action = random.randint()
        self.actions.append(action)

        return action

    def loss_function(self, logits):

        self.outputs_softmax = tf.nn.softmax(logits, name='A3')
        # self.outputs_softmax = logits
        logits = tf.Print(logits, [logits], "logits",first_n=None,summarize=50,name=None)
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits)

        loss = tf.reduce_mean(cost*self.discounted_rewards)

        # logits=tf.Print(logits,[logits[0]],'logits')

        # cost =tf.square(tf.square(self.Y-logits))
        # print('cost shape ', cost.get_shape())

        cost = tf.Print(cost, [cost], message='cost:  ', first_n=None, summarize=50, name=None)

        #
        # loss = tf.metrics.mean_squared_error(
        #     labels = self.Y,
        #     predictions=logits)

        # loss =  tf.reduce_mean(cost)
             # loss = tf.Print(loss, [loss], 'loss')
        loss = tf.Print(loss, [loss], 'cost', first_n=None, summarize=50, name=None)

        self.loss_func = loss
        self.optimiz = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def train(self):

        data_len = len(self.actions)
        disc_rewards = self.normalize(self.rewards)
        quote = np.array(self.quotes)

        quote = quote.reshape(data_len,self.output_sz, self.time_len, self.channels)
        position = np.array(self.positions)

        position = position.reshape(-1, self.nr_positions)

        actions = np.array(self.actions)
        actions = actions.reshape(data_len, self.output_sz)

        quote = quote.reshape(-1, self.nr_positions, self.time_len, self.channels)
        position = np.array(position).reshape(-1, self.nr_positions)
        # _, c = self.tf_session.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
        poli_grad = np.multiply(actions,disc_rewards[:,np.newaxis])
        # y = np.ones((poli_grad.shape))
        # y[:,0]=y[:,0]*-1.4
        # y[:, 2] = y[:, 0] * -2
        # y[:, 3] = y[:, 0] * -13
        _,c =self.tf_sess.run([self.optimiz,self.loss_func], feed_dict={self.X: quote, self.pos:position, self.Y:actions, self.discounted_rewards: disc_rewards})
        print(' COST ',c)
        print('actions ',actions[30], 'quote ',quote[30,:,-1])
        # if self.save_path is not None:
        #     save_path = self.saver.save(self.tf_sess, self.save_path)
        #     print("Model saved in file: %s" % save_path)
        self.lr *= self.lr_decay
        if self.lr < 0.0001:
            self.lr = 0.0001
        self.actions =[]
        self.positions =[]
        self.quotes =[]
        self.rewards =[]
