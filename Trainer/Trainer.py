import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import os
import utils
from data_reader import data_reader


class Trainer():
    def __init__(self, train_params):
        self.train_params = train_params

    def data_generator(self):
        data_path = '../processed_dataset/frame_step_1'
        c = data_reader(data_path)
        self.x_train, self.x_test, label_train, label_test, self.gestures = c.run()
        # self.y_train = utils.one_hot((label_train, len(self.gestures)))
        # self.y_test = utils.one_hot((label_test, len(self.gestures)))
        self.y_train = label_train
        self.y_test = label_test
        print('x_train:', np.shape(self.x_train))
        print('x_test:', np.shape(self.x_test))
        print('y_train:', np.shape(self.y_train))
        print('y_test:', np.shape(self.y_test))

    def parameters(self):
        print(self.train_params.values())
        (self.learning_rate, self.lambda_loss_amount, self.ext_epochs, self.batch_size, self.n_hidden, self.display_iter
         ) = self.train_params.values()
        self.model_name = 'HG_model_'

        training_data_count = len(self.x_train)
        test_data_count = len(self.x_test)
        self.n_steps = self.x_train.shape[1]
        self.n_input = self.x_train.shape[2]
        print('n_steps', self.n_steps, 'n_input', self.n_input)
        self.training_iters = training_data_count*self.ext_epochs # Loop ext_epochs times on the dataset
        self.n_classes = len(self.gestures)
        # self.display_iter # To show test set accuracy during training

    def build_network(self):
        # Graph input/output
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])

        # Graph weights
        weights = {
            'hidden': tf.Variable(tf.random.normal([self.n_input, self.n_hidden])),  # Hidden layer weights
            'out': tf.Variable(tf.random.normal([self.n_hidden, self.n_classes], mean=1.0))
        }
        biases = {
            'hidden': tf.Variable(tf.random.normal([self.n_hidden])),
            'out': tf.Variable(tf.random.normal([self.n_classes]))
        }

        self.pred = utils.LSTM_RNN(self.x, weights, biases, self.n_input, self.n_steps, self.n_hidden)

        # Loss, optimizer and evaluation
        l2 = self.lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        )  # L2 loss prevents this overkill neural network to overfit the data
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.pred)) + l2  # Softmax loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)  # Adam Optimizer

        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self):
        # To keep track of training's performance
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []

        # Launch the graph
        sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)

        # Perform Training steps with "batch_size" amount of example data at each loop
        step = 1
        while step * self.batch_size <= self.training_iters:
            batch_xs = utils.extract_batch_size(self.x_train, step, self.batch_size)
            batch_ys = utils.one_hot(utils.extract_batch_size(self.y_train, step, self.batch_size), self.n_classes)

            # Fit training using batch data
            _, loss, acc = sess.run(
                [self.optimizer, self.cost, self.accuracy],
                feed_dict={
                    self.x: batch_xs,
                    self.y: batch_ys
                }
            )
            train_losses.append(loss)
            train_accuracies.append(acc)

            # Evaluate network only at some steps for faster training:
            if (step * self.batch_size % self.display_iter == 0) or (step == 1) or (step * self.batch_size > self.training_iters):
                # To not spam console, show training accuracy/loss in this "if"
                print("Training iter #" + str(step * self.batch_size) + \
                      ":   Batch Loss = " + "{:.6f}".format(loss) + \
                      ", Accuracy = {}".format(acc))

                # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
                loss, acc = sess.run(
                    [self.cost, self.accuracy],
                    feed_dict={
                        self.x: self.x_test,
                        self.y: utils.one_hot(self.y_test, self.n_classes)
                    }
                )
                test_losses.append(loss)
                test_accuracies.append(acc)
                print("PERFORMANCE ON TEST SET: " + \
                      "Batch Loss = {}".format(loss) + \
                      ", Accuracy = {}".format(acc))

            step += 1

        print("Optimization Finished!")

        # Accuracy for test data

        one_hot_predictions, accuracy, final_loss = sess.run(
            [self.pred, self.accuracy, self.cost],
            feed_dict={
                self.x: self.x_test,
                self.y: utils.one_hot(self.y_test)
            }
        )

        test_losses.append(final_loss)
        test_accuracies.append(accuracy)

        print("FINAL RESULT: " + \
              "Batch Loss = {}".format(final_loss) + \
              ", Accuracy = {}".format(accuracy))

    def run(self):
        self.data_generator()
        self.parameters()
        self.build_network()
        self.train()






if __name__ == '__main__':
    train_params = {'Learning_rate': 1e-3, 'lamda_loss_amount': 0.0015, 'ext_epochs': 800, 'batch_size': 128,
                    'n_hidden': 32, 'display_iter': 30000}
    c = Trainer(train_params)
    c.run()