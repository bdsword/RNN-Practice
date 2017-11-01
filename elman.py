#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

"""
inputs:

   00111001
+  01000111
-------------
   10000000


inputs:
1 0 0 1 1 1 0 0
1 1 1 0 0 0 1 0

labels:
0 0 0 0 0 0 0 1

"""


def main(_):
    input_dim = 2
    output_dim = 1
    time_length = 8
    hidden_dim = 20
    batch_size = 5

    inputs = tf.placeholder(dtype=tf.float32,
                            shape=[time_length, batch_size , input_dim])
    labels = tf.placeholder(dtype=tf.float32,
                            shape=[time_length, batch_size, output_dim])

    test_inputs = tf.placeholder(dtype=tf.float32,
                                 shape=[time_length, 1, input_dim])

    with tf.device('/gpu:0'):
        weights = {
            'i_h': tf.Variable(tf.random_normal([input_dim + hidden_dim, hidden_dim])),
            'h_o': tf.Variable(tf.random_normal([hidden_dim, output_dim]))
        }
        biases = {
            'i_h': tf.Variable(tf.zeros([hidden_dim])),
            'h_o': tf.Variable(tf.zeros([output_dim]))
        }
        memory = tf.Variable(tf.zeros([batch_size, hidden_dim]))

    loss_op = tf.Variable(initial_value=0.0)
    
    for i in range(time_length):
        concat_input = tf.concat([inputs[i], memory], 1)
        logits = tf.add(tf.matmul(concat_input, weights['i_h']), biases['i_h'])
        memory = activated = tf.sigmoid(logits)
        out_logits = tf.add(tf.matmul(activated, weights['h_o']), biases['h_o'])
        
        loss_op += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[i],
                                                    logits=out_logits)
        )
    
    test_memory = tf.Variable(tf.zeros([1, hidden_dim]), trainable=False)
    test_outputs = []
    for i in range(time_length):
        test_concat_input = tf.concat([test_inputs[i], test_memory], 1)
        test_logits = tf.add(tf.matmul(test_concat_input, weights['i_h']), biases['i_h'])
        test_memory = test_activated = tf.sigmoid(test_logits)
        test_out_logits = tf.add(tf.matmul(test_activated, weights['h_o']), biases['h_o'])
        test_outputs.append(tf.sigmoid(test_out_logits))
        

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(loss_op)

    init = tf.global_variables_initializer()


    data = np.load(open('data.npz', 'rb'))
    train_samples = data['samples']
    train_labels = data['labels']
    
    with tf.Session() as sess:
        sess.run(init)
        num_epoch = 10000

        for cur_epoch in range(num_epoch):
            total_loss = 0
            step_rounds = np.shape(train_samples)[1] // batch_size
            for i in range(step_rounds):
                cur_samples = train_samples[:, i * batch_size : (i+1) * batch_size, :]
                cur_labels = train_labels[:, i * batch_size : (i+1) * batch_size]
                _, loss = sess.run([train_op, loss_op], {inputs: cur_samples, labels: cur_labels})
                total_loss += loss
            total_loss /= step_rounds
            print('loss: {}'.format(total_loss))

        # 01001111
        # 01001101
        #=10011100
        test_input = [[[1, 1]], [[1, 0]], [[1, 1]], [[1, 1]], [[0, 0]], [[0, 0]], [[1, 1]], [[0, 0]]]
        gg = sess.run(test_outputs, {test_inputs: test_input})
        print(gg)


if __name__ == '__main__':
    tf.app.run()
