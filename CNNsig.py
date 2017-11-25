# coding: utf-8
import tensorflow as tf

class sigGF: 
    def __init__(self, image_shape, person_num):
        self.image_shape = image_shape
        self.person_num = person_num
        
        self.X = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])
        self.Y_M = tf.placeholder(tf.float32, [None, person_num])
        self.Y_D = tf.placeholder(tf.float32, [None, 2])
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        
        self.feature_vec = None
        self.cost_M = None
        self.cost_D = None
        self.logit_M = None
        self.logit_D = None
        self.optimizer = None      
        
        self.build_model()      
        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        
        _input = self.X
        W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
        L1 = tf.nn.conv2d(_input, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)

        L1 = tf.nn.max_pool(L1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

        L3 = tf.contrib.layers.flatten(L2)
        L3 = tf.layers.dense(L3, 512, activation=tf.nn.relu)
        L3 = tf.layers.dropout(L3, self.keep_prob)

        L4 = tf.contrib.layers.flatten(L3)
        L4 = tf.layers.dense(L4, 256, activation=tf.nn.relu)
        L4 = tf.layers.dropout(L4, self.keep_prob)
        
        self.feature_vec = L4
        
        self.logit_M = tf.layers.dense(L4, self.person_num, activation=None, trainable = False)
        self.cost_M = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logit_M, labels=self.Y_M))
        self.logit_D = tf.layers.dense(L4, 2, activation=None, trainable = False)
        self.cost_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logit_D, labels=self.Y_D))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost_M + self.cost_D)
        
    def train(self, session, X, Y_M, Y_D):
        return session.run([self.optimizer, self.cost_M, self.cost_D], feed_dict = {self.X: X,
                                                                                    self.Y_M: Y_M,
                                                                                    self.Y_D: Y_D,
                                                                                    self.keep_prob: 0.7})
    def predict(self, session, X):
        return session.run([self.logit_M, self.logit_D], feed_dict = {self.X: X, self.keep_prob: 1})
                
    def test(self, session, X, Y_M, Y_D):
        self.is_correct = tf.equal(tf.argmax(self.logit_M, 1), tf.argmax(self.Y_M, 1)) 
        self.acc_M = tf.reduce_mean(tf.cast(self.is_correct, tf.float32)) 
                
        self.is_correct2 = tf.equal(tf.argmax(self.logit_D, 1), tf.argmax(self.Y_D, 1)) 
        self.acc_D = tf.reduce_mean(tf.cast(self.is_correct2, tf.float32))
        
        return session.run([self.acc_M, self.acc_D], feed_dict={self.X: X,
                                                          self.Y_M: Y_M,
                                                          self.Y_D: Y_D,
                                                          self.keep_prob: 1})
    def get_feature(self, session, X):
        return session.run(self.feature_vec, feed_dict = {self.X: X, self.keep_prob: 1})

class sigG: 
    def __init__(self, image_shape, person_num):
        self.image_shape = image_shape
        self.person_num = person_num
        
        self.X = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])
        self.Y_M = tf.placeholder(tf.float32, [None, person_num])
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        
        self.feature_vec = None
        self.cost_M = None
        self.logit_M = None
        self.optimizer = None
        
        self.build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        
        _input = self.X
        W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
        L1 = tf.nn.conv2d(_input, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)

        L1 = tf.nn.max_pool(L1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

        L3 = tf.contrib.layers.flatten(L2)
        L3 = tf.layers.dense(L3, 512, activation=tf.nn.relu)
        L3 = tf.layers.dropout(L3, self.keep_prob)

        L4 = tf.contrib.layers.flatten(L3)
        L4 = tf.layers.dense(L4, 256, activation=tf.nn.relu)
        L4 = tf.layers.dropout(L4, self.keep_prob)
        
        self.feature_vec = L4
        
        self.logit_M = tf.layers.dense(L4, self.person_num, activation=None, trainable = False)
        self.cost_M = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logit_M, labels=self.Y_M))

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost_M)
        
    def train(self, session, X, Y_M):
        return session.run([self.optimizer, self.cost_M], feed_dict = {self.X: X,self.Y_M: Y_M,
                                                                       self.keep_prob: 0.7})
    def predict(self, session, X):
        return session.run(self.logit_M, feed_dict = {self.X: X, self.keep_prob: 1})
                
    def test(self, session, X, Y_M):
        self.is_correct = tf.equal(tf.argmax(self.logit_M, 1), tf.argmax(self.Y_M, 1)) 
        self.acc_M = tf.reduce_mean(tf.cast(self.is_correct, tf.float32)) 
        return session.run(self.acc_M, feed_dict={self.X: X, self.Y_M: Y_M, self.keep_prob: 1})
    
    def get_feature(self, session, X):
        return session.run(self.feature_vec, feed_dict = {self.X: X, self.keep_prob: 1})