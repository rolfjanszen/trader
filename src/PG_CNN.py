from policygrad import PolicyGrad
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import os
import random
class LSTMPolicyGrad(PolicyGrad):

    lr = 0.0001
    lr_decay = 0.99
    gamma = 0.98
    quotes = []
    positions= []
    rewards = []
    actions = []
    loss_func = None
    def __init__(self, input_sz_, positions, action_sz , chunk_size_, max_len_sent, rnn_size_,channels_, save_path=None, load_path=None):
        self.n_classes = action_sz
        self.time_len = max_len_sent
        self.chunk_size = int(chunk_size_)
        self.n_chunks = int(max_len_sent / chunk_size_)
        self.rnn_size = rnn_size_
        # self.X = tf.placeholder('float', [None, self.n_chunks, self.chunk_size])
        self.nr_positions = positions
        self.output_sz = action_sz
        self.channels = channels_
        self.X = tf.placeholder('float', [None,positions, max_len_sent,self.channels])
        self.Y = tf.placeholder('float', [None, self.output_sz])
        self.training = tf.placeholder('bool',None)
        self.pos = tf.placeholder('float', [None, self.output_sz])
        self.discounted_rewards = tf.placeholder(tf.float32, [None, ], name="actions_value")
        self.input_sz = input_sz_


        logits = self.model_cnn(self.X,self.pos,self.training)
        # logits = self.lstm_model(self.X)
        self.loss_function(logits)

        self.tf_sess.run(tf.global_variables_initializer())
        self.save_path = None

        self.saver = tf.train.Saver()

        if save_path is not None and os.path.isfile(os.path.abspath(save_path + '.meta')):

            self.load_path = save_path
            try:
                self.saver.restore(self.tf_sess, self.load_path)
                print('model loaded')
            except:
                print('could not reload weights')

        if save_path is not None:
            self.save_path = save_path

    def model_cnn(self, x, pos,training):
        # x*=100
        filter1 = tf.Variable(tf.random_normal([1,9,self.channels,4]))
        filter2 = tf.Variable(tf.random_normal([1, 3,4,16]))
        filter3 = tf.Variable(tf.random_normal([1, 6, 16, 32]))
        filter4 = tf.Variable(tf.random_normal([1, 3,32, 64]))
        filter5 = tf.Variable(tf.random_normal([1, self.time_len - 17, 64, 64]))
        # x = tf.Print(x, [x], "x")
        # pos = tf.Print(pos, [pos], "pos")
        # filter1 = tf.Print(filter1, [filter1], "layer_1")
        # filter2 = tf.Print(filter2, [filter2], "filter2")

        use_batch_norm = False #Batch norm doesn't help. At all!
        layer_1 = tf.nn.conv2d(x,filter1,[1,1,1,1],'VALID')
        if use_batch_norm:
            layer_1 = tf.contrib.layers.batch_norm(layer_1, data_format='NHWC',  center=True,scale=True,is_training=training)
        layer_1 = tf.nn.tanh(layer_1)

        layer_2 = tf.nn.conv2d(layer_1, filter2, [1, 1, 1, 1], 'VALID')
        if use_batch_norm:
            layer_2 = tf.contrib.layers.batch_norm(layer_2,data_format='NHWC',  center=True,scale=True,is_training=training)
        layer_2 = tf.nn.tanh(layer_2)
        layer_3 = tf.nn.conv2d(layer_2, filter3, [1, 1, 1, 1], 'VALID')
        if use_batch_norm:
            layer_3 = tf.contrib.layers.batch_norm(layer_3, data_format='NHWC', center=True, scale=True, is_training=training)
        layer_3 = tf.nn.tanh(layer_3)

        layer_4 = tf.nn.conv2d(layer_3, filter4, [1, 1, 1, 1], 'VALID')
        if use_batch_norm:
            layer_4 = tf.contrib.layers.batch_norm(layer_4, data_format='NHWC', center=True, scale=True, is_training=training)
        layer_4 = tf.nn.tanh(layer_4)

        layer_5 = tf.nn.conv2d(layer_4, filter5, [1, 1, 1, 1], 'VALID')
        if use_batch_norm:
            layer_5 = tf.contrib.layers.batch_norm(layer_5, data_format='NHWC', center=True, scale=True, is_training=training)
        layer_5 = tf.nn.tanh(layer_5)
        dims_5 = layer_5.get_shape().as_list()
        total_size = np.prod(dims_5[1:])
        print(' total_size cnn ',total_size)
        flat_tensor = tf.reshape(layer_5, [-1,total_size])

        combined = tf.concat((flat_tensor, pos),1)
        # print('combined sh ',combined.shape)
        # combined = tf.Print(combined, [combined], 'combined')
        # output = tf.contrib.layers.fully_connected(flat_tensor,  self.output_sz)
        # output = output/tf.reduce_sum(output)

        fc_1 = tf.layers.dense(inputs=combined,
                                 units=10,
                                 activation=None)
        fc_1 = tf.nn.tanh(fc_1)
        output = tf.layers.dense(inputs=fc_1,
                                 units=self.output_sz,
                                 activation=None)
        # output = tf.Print(output, [output], "output")
        print('output ', output.shape)

        return output


    def lstm_model(self, x):
        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, 4])),
                 'biases': tf.Variable(tf.random_normal([4]))}

        x = tf.transpose(x, [2, 0, 1,3])
        x = tf.reshape(x, [-1, self.chunk_size])
        x = tf.split(x,self. n_chunks, 0)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        outputs =tf.reshape(outputs,[-1])
        output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

        return output


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
        position = np.array(position).reshape(-1, self.output_sz)
        prob_weights =self.tf_sess.run(self.outputs_softmax, feed_dict={self.X: quote, self.pos: position,self.training:False})


        sum_prob = 1
        if sum_prob < 0.000001:
            action = [0.2,0.4,0.1,0.30]
            print('ZERO SUM PROBABILITY DETECTED')
        else:
            action = prob_weights[0] /sum_prob #make sure it sums to 1

        action = np.round(action,2)
        self.actions.append(action)

        return action

    def loss_function(self, logits):

        self.outputs_softmax = tf.nn.softmax(logits, name='A3')
        # self.outputs_softmax = logits
        logits = tf.Print(logits, [logits], "logits",first_n=None,summarize=50,name=None)
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits)

        loss = tf.reduce_mean(cost*self.discounted_rewards)
        self.loss_func = loss
        self.optimiz = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def train(self):
        self.print_performance()
        data_len = len(self.actions)
        disc_rewards = self.normalize(self.rewards)
        # disc_rewards=self.rewards
        quote = np.array(self.quotes)

        quote = quote.reshape(data_len,self.nr_positions, self.time_len, self.channels)
        position = np.array(self.positions).reshape(-1, self.output_sz)

        actions = np.array(self.actions)
        actions = actions.reshape(data_len, self.output_sz)

        batch_sz = 300
        total_sz = len(actions)
        batch_start = 0
        for batch_end in range(batch_sz, total_sz, batch_sz):

            batch_quote = quote[batch_start:batch_end]
            batch_position = position[batch_start:batch_end]
            batch_actions = actions[batch_start:batch_end]
            batch_disc_rewards = disc_rewards[batch_start:batch_end]
            _,c =self.tf_sess.run([self.optimiz,self.loss_func], feed_dict={self.X: batch_quote, self.pos:batch_position, self.Y:batch_actions, self.discounted_rewards:-batch_disc_rewards,self.training :True})
            print('cost ', c)
            batch_start = batch_end


        if self.save_path is not None:
            save_path = self.saver.save(self.tf_sess, self.save_path)
            print("Model saved in file: %s" % save_path)

        self.lr *= self.lr_decay
        if self.lr < 0.00001:
            self.lr = 0.00001
        self.actions =[]
        self.positions =[]
        self.quotes =[]
        self.rewards =[]



    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def print_performance(self):
        softm_sum = [0]*len(self.actions[0])
        action_sum = [0] * len(self.actions[0])
        last_action =[0]*len(self.actions[0])
        entropy_action = [0] * len(self.actions[0])
        for action in self.actions:
            softm_sum += self.softmax(action)
            action_sum += action
            change_action = last_action - action
            entropy_action += np.abs(change_action)
            last_action = action

        print('softM ',softm_sum/np.mean(softm_sum), ' action_sum ',action_sum/np.mean(action_sum), ' entropy_action ',entropy_action/np.mean(entropy_action))
