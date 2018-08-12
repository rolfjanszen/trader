import tensorflow as tf  

from tensorflow.python.ops import rnn, rnn_cell

import numpy as np 

class RnnModel():
    # n_classes = 0
    # chunk_size = 0
    # n_chunks = 0
    # rnn_size = 0

    save_file = 'model.ckpt'
    prediction =1
    saver = None
    tf_session = tf.Session()
    lr_decay =0.99
    lr =0.002
    def __init__(self, chunk_size_, max_len_sent, rnn_size_, classes):
        self.n_classes = classes
        self.chunk_size = int(chunk_size_)
        self.n_chunks = int(max_len_sent/chunk_size_)
        self.rnn_size = rnn_size_
        self.x = tf.placeholder('float', [None, self.chunk_size, self.n_chunks,1])
        # self.x = tf.placeholder('float', [None, self.n_chunks, self.chunk_size])
        self.y = tf.placeholder('float', [None,classes])
        self.prediction = self.cnn_model(self.x)


    def cnn_model(self, x):

        filter1 = tf.Variable(tf.random_normal([1,3,1,4]))
        filter2 = tf.Variable(tf.random_normal([1,9,4,4]))
        filter3 = tf.Variable(tf.random_normal([1,36,4,16]))

        layer1 = tf.nn.conv2d(x, filter1,[1,1,1,1],'VALID')
        layer1 = tf.nn.relu(layer1)
        print('layer 1 ', layer1.shape)
        layer2 = tf.nn.conv2d( layer1, filter2, [1, 1, 1, 1], 'VALID')
        layer2 = tf.nn.relu(layer2)
        print('layer 2 ', layer2.shape)
        layer3 = tf.nn.conv2d(layer2, filter3, [1, 1, 1, 1], 'VALID')
        layer3 = tf.contrib.layers.flatten(layer3)

        output = tf.layers.dense(inputs = layer3,
                                 units = self.n_classes,
                                 activation = None )

        return output



    def simple_model(self, x):
        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                 'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.chunk_size])
        x = tf.split(x,self. n_chunks, 0)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

        return output

    def recurrent_neural_network(self, x,drop):
        weights_1 = {'weights':tf.Variable(tf.random_normal([self.rnn_size, 150])),
                 'biases':tf.Variable(tf.random_normal([150]))}
        
        weights_2 = {'weights':tf.Variable(tf.random_normal([150,50])),
                 'biases':tf.Variable(tf.random_normal([50]))}
        weights_3 = {'weights':tf.Variable(tf.random_normal([50,self.n_classes])),
                 'biases':tf.Variable(tf.random_normal([self.n_classes]))}
            
        x = tf.transpose(x, [1,0,2])
        x = tf.reshape(x, [-1, self.chunk_size])
        x = tf.split(x, self.n_chunks, 0)
        
        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size,state_is_tuple=True)
        lstm_cell_drop = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,input_keep_prob=drop, output_keep_prob=drop,state_keep_prob=drop)
    
        outputs, states = rnn.static_rnn(lstm_cell_drop, x, dtype=tf.float32)
    #     lstm_cell2 = rnn_cell.BasicLSTMCell(self.rnn_size,state_is_tuple=True)
#         outputs2, states = rnn.static_rnn(lstm_cell, outputs, dtype=tf.float32)
        outputs2 = tf.nn.dropout(outputs, drop)
        layer1 = tf.nn.relu( tf.matmul(outputs2[-1],weights_1['weights']) + weights_1['biases'])
        layer1 = tf.nn.dropout(layer1, drop)
        layer2 = tf.nn.relu6( tf.matmul(layer1,weights_2['weights']) + weights_2['biases'])
        layer2 = tf.nn.dropout(layer2, drop)
        layer3 = tf.matmul(layer2,weights_3['weights'])
        
        return layer3
    
    def train_neural_network(self,data_in,output_y):
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.prediction ))
        cost = tf.reduce_mean(tf.square(self.y - self.prediction))
        optimizer = tf.train.AdamOptimizer(0.002).minimize(cost)
        init = tf.global_variables_initializer()
        data_size = len(data_in)
        batch_size = 100
        hm_epochs = 2000
        # saver = tf.train.Saver()
        # with self.tf_session as sess:
        self.tf_session.run(init)
#             saver = tf.train.Saver()

        # saver.restore(sess, self.save_file)
        # saver.save(sess, self.save_file)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            start_batch = 0
            self.lr = self.lr*self.lr_decay
            for end_batch in range(batch_size,data_size,batch_size):
                epoch_x = np.array(data_in[start_batch:end_batch])
                epoch_y = output_y[start_batch:end_batch]

                epoch_x = epoch_x.reshape((batch_size,  self.chunk_size,self.n_chunks, 1))

                _, c = self.tf_session.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
                epoch_loss += c
                start_batch = end_batch

            print('loss ',epoch_loss)
            # saver.save(sess, self.save_file)
                

    def make_stock_prediction(self, input_data):
        tf.reset_default_graph()
        batch_size = 100
        predicted =[0]*100
        data_size = len(input_data)

        start_batch = 0

        for end_batch in range(batch_size, data_size, batch_size):
            epoch_x = np.array(input_data[start_batch:end_batch])
            epoch_x = epoch_x.reshape((batch_size,  self.chunk_size,self.n_chunks,1))

            start_batch = end_batch
            results = self.tf_session.run(self.prediction, feed_dict={self.x:epoch_x})
            for res in results:
                print(res)
                predicted.append(res[0])
#
        return predicted
    
           
           
           
           
