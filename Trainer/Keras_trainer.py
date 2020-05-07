import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import os
import utils
import tools
import models_training
from data_reader import data_reader
# 4 generation: Quantized weight, State Saturation aware training.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import time
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from datetime import datetime


class Trainer():
    def __init__(self, train_params):
        self.train_params = train_params
        self.time_window = 20 # smaller than 20

    def data_generator(self):
        data_path = '../processed_dataset/frame_step_1_seperate_no_empty'
        c = data_reader(data_path, self.time_window)
        self.x_train, self.x_test, label_train, label_test, self.gestures = c.run()
        self.y_train = utils.one_hot(label_train, len(self.gestures))
        self.y_test = utils.one_hot(label_test, len(self.gestures))
        self.n_classes = len(self.gestures)
        print('x_train:', np.shape(self.x_train))
        print('x_test:', np.shape(self.x_test))
        print('y_train:', np.shape(self.y_train))
        print('y_test:', np.shape(self.y_test))

    def parameters(self):
        print(self.train_params.values())
        (self.learning_rate, self.lambda_loss_amount, self.ext_epochs, self.batch_size, self.save) = self.train_params.values()
        self.model_name = 'KTH_model_'
        self.data_gen_step = self.batch_size / 32
        self.save_dir = os.path.join(self.save, str(self.learning_rate) + '_' + str(self.batch_size) + '_' + str(self.time_window))

    def optimizer(self):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=tools.lr_schedule(0))
        adam = Adam(lr=self.learning_rate)
        return adam

    def callbacks(self):
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        return tensorboard_callback

    def tf_gpus_options(self):
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        print(gpus, cpus)
        ### By default, TensorFlow requests and blocks nearly all of the GPU memory of the two available GPUs to avoid memory fragmentation
        ### Ask for memory only when you needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        ### Limit the consumption of memory use to 2GB
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        # )
        ### Create two virtual gpu in one GPU
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        #print(tf.config.get_soft_device_placement)
        print(tf.config.experimental_list_devices())

    def run(self):
        self.tf_gpus_options()
        self.parameters()
        self.data_generator()
        print(self.x_train.shape[1:])
        model = models_training.HG_model(input_shape=self.x_train.shape[1:], num_classes=self.n_classes, r=1e-3)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer(), metrics=['accuracy'])
        cbs = tools.monitoring(self.save_dir, self.batch_size, self.model_name)
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size,  epochs=self.ext_epochs, shuffle=True,
                  validation_data=[self.x_test, self.y_test],callbacks=cbs)
#tools.monitoring(self.save_dir, self.batch_size, self.model_name)

if __name__ == '__main__':
    train_params = {'Learning_rate': 1e-3, 'lamda_loss_amount': 0.0015, 'ext_epochs': 800, 'batch_size': 64, 'save_dir': 'models/frame_step_1_seperate_no_empty'}
    c = Trainer(train_params)
    c.run()