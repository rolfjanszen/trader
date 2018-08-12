import tensorflow as tf
import numpy as np
from abc import abstractmethod

class PolicyGrad:
    input_sz = 1
    output_sz = 1
    tf_sess = tf.Session()
    optimiz = None
    cost = None
    X = None
    Y = None
    discounted_rewards = None

    lr = 0.002
    gamma = 0.99

    def __init__(self, input_sz_, output_sz_, save_path=None, load_path=None):
        self.input_sz = input_sz_
        self.output_sz = output_sz_

        self.X = tf.placeholder(tf.float32, shape=(self.input_sz, None), name="X")
        self.Y = tf.placeholder(tf.float32, shape=(self.output_sz, None), name="Y")
        self.discounted_rewards = tf.placeholder(tf.float32, [None, ], name="actions_value")

        logits = self.model(self.X)
        self.loss_function(logits)

        self.tf_sess.run(tf.global_variables_initializer())
        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

    #         if load_path is not None:
    #             self.load_path = load_path
    #             self.saver.restore(self.tf_sess, self.load_path)
    @abstractmethod
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

    @abstractmethod
    def get_action(self, observation):

        prob_weights = self.tf_sess.run(self.outputs_softmax, feed_dict={self.X: observation[:, np.newaxis]})

        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        # print('action ', action, prob_weights)
        return action

    @abstractmethod
    def model(self, x):

        layer_1 = {'weights':tf.Variable(tf.random_normal([10,self.input_sz])),
                   'biases':tf.Variable(tf.random_normal([10,1]))}


        layer_2 = {'weights':tf.Variable(tf.random_normal([10,10])),
                   'biases':tf.Variable(tf.random_normal([10,1]))}


        layer_3 = {'weights':tf.Variable(tf.random_normal([self.output_sz,10])),
                   'biases':tf.Variable(tf.random_normal([self.output_sz,1]))}
        #         tf.layers.dropout


        tf.matmul(layer_1['weights'],x)
        l1 = tf.add(tf.matmul(layer_1['weights'],x),layer_1['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(layer_2['weights'],l1),layer_2['biases'])
        l2 = tf.nn.sigmoid(l2)

        l3 = tf.add(tf.matmul(layer_3['weights'],l2),layer_3['biases'])
        logits = tf.transpose(l3)
        return logits
        # units_layer_1 = 10
        # units_layer_2 = 10
        #
        # with tf.name_scope('parameters'):
        #     W1 = tf.get_variable("W1", [units_layer_1, self.input_sz],
        #                          initializer=tf.contrib.layers.xavier_initializer(seed=1))
        #     b1 = tf.get_variable("b1", [units_layer_1, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        #     W2 = tf.get_variable("W2", [units_layer_2, units_layer_1],
        #                          initializer=tf.contrib.layers.xavier_initializer(seed=1))
        #     b2 = tf.get_variable("b2", [units_layer_2, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        #     W3 = tf.get_variable("W3", [self.output_sz, units_layer_2],
        #                          initializer=tf.contrib.layers.xavier_initializer(seed=1))
        #     b3 = tf.get_variable("b3", [self.output_sz, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        #
        # # Forward prop
        # with tf.name_scope('layer_1'):
        #     Z1 = tf.add(tf.matmul(W1, self.X), b1)
        #     A1 = tf.nn.relu(Z1)
        # with tf.name_scope('layer_2'):
        #     Z2 = tf.add(tf.matmul(W2, A1), b2)
        #     A2 = tf.nn.relu(Z2)
        # with tf.name_scope('layer_3'):
        #     Z3 = tf.add(tf.matmul(W3, A2), b3)
        #     A3 = tf.nn.softmax(Z3)
        # logits = tf.transpose(Z3)


    def loss_function(self, logits):

        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        #         pred = self.model(l3)
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(self.Y), logits=logits)
        loss = tf.reduce_mean(cost * self.discounted_rewards)
        self.optimiz = tf.train.AdamOptimizer(self.lr).minimize(loss)

    @abstractmethod
    def train(self, state, actions, rewards):

        disc_rewards = self.normalize(rewards)
        self.tf_sess.run(self.optimiz, feed_dict={self.X: np.vstack(state).T, self.Y: np.vstack(actions).T,  self.discounted_rewards: disc_rewards})

        if self.save_path is not None:
            save_path = self.saver.save(self.tf_sess, self.save_path)
            print("Model saved in file: %s" % save_path)


#         epoch_loss += c

