from policygrad import PolicyGrad
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import os
import random
class CNNPolicyGrad(PolicyGrad):

    lr = 0.001
    lr_decay = 0.99
    gamma = 0.98
    drop_rate = 0.1
    alpha =0.00001
    quotes = []
    positions= []
    rewards = []
    actions = []
    loss_func = None

    def __init__(self, input_sz_, positions, action_sz , chunk_size_, max_len_sent, rnn_size_,channels_, save_path=None, name_ = ''):
        self.n_classes = action_sz
        self.time_len = max_len_sent
        self.chunk_size = int(chunk_size_)
        self.n_chunks = int(max_len_sent / chunk_size_)
        self.rnn_size = rnn_size_
        # self.X = tf.placeholder('float', [None, self.n_chunks, self.chunk_size])
        self.nr_positions = positions
        self.output_sz = action_sz
        self.channels = channels_
        self.name = name_

        self.X = tf.placeholder('float', [None,positions, max_len_sent,self.channels])
        self.Y = tf.placeholder('float', [None, self.output_sz])
        self.training = tf.placeholder('bool',None)
        self.pos = tf.placeholder('float', [None, self.output_sz])
        self.discounted_rewards = tf.placeholder(tf.float32, [None, ], name="actions_value")
        self.input_sz = input_sz_

        logits = self.model(self.X,self.pos,self.training)
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

    def cnn_model(self, x, is_training):
        with tf.variable_scope("cnn_kernels"):
            filter1 = tf.Variable(tf.random_normal([1,3,self.channels,16]), name="kernel_1")
            filter2 = tf.Variable(tf.random_normal([1, 3,16,16]), name="kernel_2")
            filter3 = tf.Variable(tf.random_normal([1, 3, 16, 32]), name="kernel_3")
            filter4 = tf.Variable(tf.random_normal([1, 3,32, 64]), name="kernel_4")
            filter5 = tf.Variable(tf.random_normal([1, self.time_len - 10, 64, 64]), name="kernel_5")
        # x = tf.Print(x, [x], "x")
        # pos = tf.Print(pos, [pos], "pos")
        # filter1 = tf.Print(filter1, [filter1], "layer_1")
        # filter2 = tf.Print(filter2, [filter2], "filter2")

        use_batch_norm = False #Batch norm doesn't help. At all!
        with tf.name_scope('convolutions'+self.name) as scope:

            layer_1 = tf.nn.conv2d(x,filter1,[1,1,1,1],'VALID',name="cnn_layer_1")
            if use_batch_norm:
                layer_1 = tf.contrib.layers.batch_norm(layer_1, data_format='NHWC',  center=True,scale=True,is_training=training)
            layer_1 = tf.layers.dropout(layer_1,rate = self.drop_rate,seed = 232, training = is_training)
            layer_1 = tf.nn.tanh(layer_1)

            layer_2 = tf.nn.conv2d(layer_1, filter2, [1, 1, 1, 1], 'VALID',name="cnn_layer_2")
            if use_batch_norm:
                layer_2 = tf.contrib.layers.batch_norm(layer_2,data_format='NHWC',  center=True,scale=True,is_training=training)
            layer_2 = tf.layers.dropout(layer_2, rate=self.drop_rate, seed=232, training = is_training)
            layer_2 = tf.nn.tanh(layer_2)

            layer_3 = tf.nn.conv2d(layer_2, filter3, [1, 1, 1, 1], 'VALID',name="cnn_layer_3")
            if use_batch_norm:
                layer_3 = tf.contrib.layers.batch_norm(layer_3, data_format='NHWC', center=True, scale=True, is_training=training)
            layer_3 = tf.layers.dropout(layer_3, rate=self.drop_rate, seed=232, training = is_training)
            layer_3 = tf.nn.tanh(layer_3)

            layer_4 = tf.nn.conv2d(layer_3, filter4, [1, 1, 1, 1], 'VALID',name="cnn_layer_4")
            if use_batch_norm:
                layer_4 = tf.contrib.layers.batch_norm(layer_4, data_format='NHWC', center=True, scale=True, is_training=training)
            layer_4 = tf.layers.dropout(layer_4, rate=self.drop_rate, seed=232, training = is_training)
            layer_4 = tf.nn.tanh(layer_4)

            layer_5 = tf.nn.conv2d(layer_4, filter5, [1, 1, 1, 1], 'VALID',name="cnn_layer_5")
            if use_batch_norm:
                layer_5 = tf.contrib.layers.batch_norm(layer_5, data_format='NHWC', center=True, scale=True, is_training=training)
            layer_5 = tf.layers.dropout(layer_5, rate=self.drop_rate, seed=232, training=is_training)
            layer_5 = tf.nn.tanh(layer_5)

        dims_5 = layer_5.get_shape().as_list()
        total_size = np.prod(dims_5[1:])
        flat_tensor = tf.reshape(layer_5, [-1,total_size])

        return flat_tensor



    def lstm_model(self, x, is_training):


        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, 4])),
                 'biases': tf.Variable(tf.random_normal([4]))}
        outputs = []
        x = tf.transpose(x, [1, 0, 2, 3])
        x_ = tf.unstack(x)
        lstm_cells = []

        i=0

        for x_entry in x_:
            # x_hold = tf.transpose(x_entry[:,2], [2, 0, 1])
            x_hold = tf.reshape(x_entry[:,:,2], [-1, self.chunk_size])
            x_hold = tf.split(x_hold,self. n_chunks, 0)

            scope_name = 'lstm_'+self.name+str(i)
            with tf.variable_scope(scope_name):
                lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
                lstm_cells.append(lstm_cell)
                output, states = rnn.static_rnn(lstm_cells[-1], x_hold, dtype=tf.float32)
                # outputs =tf.reshape(outputs,[-1])
                rnn_result = tf.matmul(output[-1], layer['weights']) + layer['biases']
                # rnn_result = tf.layers.dense(output[-1],4)
                tf.layers.dropout(rnn_result,self.drop_rate, seed=232, training=is_training)
                rnn_result = tf.nn.tanh(rnn_result)
            i+=1

        return rnn_result


    def model(self, x, pos,training):

        rnn_out = self.lstm_model(x,training)
        flat_tensor = self.cnn_model(x, training)

        print('flat_tensor sh ', flat_tensor.shape)
        # combined = tf.Print(combined, [combined], 'combined')
        # output = tf.contrib.layers.fully_connected(flat_tensor,  self.output_sz)
        # output = output/tf.reduce_sum(output)
        with tf.name_scope('fullyconnected_'+self.name) as scope:
            fc_cnn = tf.layers.dense(inputs=flat_tensor,
                                       units=30,
                                       activation=None, name="fc_layer_cnn"+self.name)

            fc_cnn= tf.nn.tanh(fc_cnn)
            combined = tf.concat(( rnn_out, pos), 1)
            print('combined sh ', combined.shape, ' fc_cnn ',fc_cnn.shape,' rnn_out ',rnn_out )

            fc_1 = tf.layers.dense(inputs=combined,
                                     units=10,
                                     activation=None, name="fc_layer_1"+self.name)
            fc_1 = tf.nn.tanh(fc_1)
            output = tf.layers.dense(inputs=fc_1,
                                     units=self.output_sz,
                                     activation=None, name="fc_layer_out"+self.name)
        # output = tf.Print(output, [output], "output")
        print('output ', output.shape)

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
        prob_weights = self.tf_sess.run(self.outputs_softmax, feed_dict={self.X: quote, self.pos: position,self.training:False})

        # sum_prob = 1
        # if sum_prob < 0.000001:
        #     action = [0.2,0.4,0.1,0.30]
        #     print('ZERO SUM PROBABILITY DETECTED')
        # else:
        #     action = prob_weights[0] / sum_prob  # make sure it sums to 1

        action = np.round(prob_weights[0],2)
        self.actions.append(action)

        return action

    def loss_function(self, logits):

        self.outputs_softmax = tf.nn.softmax(logits, name='A3')
        # self.outputs_softmax = logits
        logits = tf.Print(logits, [logits], "logits",first_n=None,summarize=50,name=None)
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits)
        weights = self.get_weights()
        regularization = tf.Variable(tf.zeros([1]))

        for weight in weights:

            weight = tf.Print(weight,[weight.name],'name')
            regularization += tf.reduce_sum(tf.abs(weight))
            regularization = tf.Print(regularization,[regularization],'sdf')

        loss = tf.reduce_mean(cost*self.discounted_rewards) + regularization*self.alpha
        self.loss_func = loss
        self.optimiz = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def reset(self):
        self.actions =[]
        self.positions =[]
        self.quotes =[]
        self.rewards =[]

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
            _,c =self.tf_sess.run([self.optimiz,self.loss_func], feed_dict={self.X: batch_quote, self.pos:batch_position, self.Y:batch_actions, self.discounted_rewards:batch_disc_rewards,self.training :True})
            print('cost ', c)
            batch_start = batch_end


        if self.save_path is not None:
            save_path = self.saver.save(self.tf_sess, self.save_path)
            print("Model saved in file: %s" % save_path)

        self.lr *= self.lr_decay
        if self.lr < 0.00001:
            self.lr = 0.00001
        print('new elarnign rate ', self.lr)
        self.reset()

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

    def get_weights(self):
        vars = tf.trainable_variables()
        # copy_ops = [vars[ix + len(vars) // 2].assign(var.value()) for ix, var in enumerate(vars[0:len(vars) // 2])]
        print(vars)
        return vars


    def replace_weights(self, new_weights):
        print('copying weights...')
        for i in range(len(new_weights) // 2):
            assign_op = new_weights[i + len(new_weights) // 2].assign(new_weights[i])
            self.tf_sess.run(assign_op)
        # map(lambda x: self.tf_sess.run( tf.group(*new_weights)