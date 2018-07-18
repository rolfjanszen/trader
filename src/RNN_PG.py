from policygrad import PolicyGrad
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import os


class LSTMPolicyGrad(PolicyGrad):

    lr = 0.0005
    gamma = 0.95
    quotes = []
    positions= []
    rewards = []
    actions = []

    def __init__(self, input_sz_, positions, output_sz_ , chunk_size_, max_len_sent, rnn_size_,save_path=None, load_path=None):
        self.n_classes = output_sz_
        self.chunk_size = int(chunk_size_)
        self.n_chunks = int(max_len_sent / chunk_size_)
        self.rnn_size = rnn_size_
        self.X = tf.placeholder('float', [None, self.n_chunks, self.chunk_size])
        self.Y = tf.placeholder('float', [None, self.n_classes])
        self.nr_positions = positions
        self.pos = tf.placeholder('float', [None, self.nr_positions])
        self.discounted_rewards = tf.placeholder(tf.float32, [None, ], name="actions_value")
        self.input_sz = input_sz_
        self.output_sz = output_sz_

        logits = self.model(self.X,self.pos)
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

    def model(self, x, pos):
        layer_pos = {'weights': tf.Variable(tf.random_normal([self.nr_positions, 5])),
                     'biases': tf.Variable(tf.random_normal([5]))}

        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size+5, self.n_classes])),
                 'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.chunk_size])
        x = tf.split(x,self. n_chunks, 0)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        pos_output = tf.matmul(pos, layer_pos['weights']) + layer_pos['biases']
        combined = tf.concat((outputs[-1],pos_output),1)
        output = tf.matmul(combined, layer['weights']) + layer['biases']

        return output

    def gather_data(self, state, reward):
        self.quotes.append(state['quote'])
        self.positions.append(state['position'])
        self.rewards.append(reward)


    def normalize(self, rewards):
        disc_reward = np.zeros_like(rewards)
        cummulative = 0
        for t in reversed(range(len(rewards))):
            cummulative = cummulative * self.gamma + rewards[t]
            disc_reward[t] = cummulative
        # disc_reward = disc_reward - np.mean(disc_reward)
        # disc_reward = disc_reward / np.std(disc_reward)
        print('disc_reward ',disc_reward)
        return disc_reward


    def get_action(self, observation):

        quote = observation['quote']
        position = observation['position']

        quote=(quote-quote[0])/quote[0]
        quote=quote.reshape(1, self.n_chunks, self.chunk_size)
        position = np.array(position).reshape(1, self.nr_positions)
        prob_weights = self.tf_sess.run(self.outputs_softmax, feed_dict={self.X: quote, self.pos: position})

        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        new_action = np.zeros(self.output_sz)
        #         print('new_action ',action)
        new_action[action] = 1

        self.actions.append(new_action)
        return action

    def loss_function(self, logits):

        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        #         pred = self.model(l3)
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits)
        loss = tf.reduce_mean(cost * self.discounted_rewards)
        self.optimiz = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def train(self):

        disc_rewards = self.normalize(self.rewards)
        quote = np.array(self.quotes)
        quote = (quote - quote[0]) / quote[0]
        quote = quote.reshape(-1, self.n_chunks, self.chunk_size)
        position = np.array(self.positions)

        position = position.reshape(-1, self.nr_positions)
        self.tf_sess.run(self.optimiz, feed_dict={self.X: quote, self.pos:position, self.Y: self.actions, self.discounted_rewards: disc_rewards})

        if self.save_path is not None:
            save_path = self.saver.save(self.tf_sess, self.save_path)
            print("Model saved in file: %s" % save_path)

        self.actions =[]
        self.positions =[]
        self.quotes =[]
        self.rewards =[]