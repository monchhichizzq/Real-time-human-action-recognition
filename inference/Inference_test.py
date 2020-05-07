import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import keras
import numpy as np

import time
from threading import Thread
from sklearn import metrics
from Trainer.data_reader import data_reader
from keras.models import load_model
from Trainer.tools import plot_confusion_matrix
from Trainer import utils
import tensorflow as tf

class Inference():
    def __init__(self):
        self.method = 'tw_20_ts_1_feature_dis_rotate'
        self.time_window = 20
        self.model_path = '../Trainer/models/seperate_sign_rotate/0.001_128_20/HG-KTH_model_-0.86_.h5'

    def tf_gpus_options(self):
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        print(gpus, cpus)
        ### By default, TensorFlow requests and blocks nearly all of the GPU memory of the two available GPUs to avoid memory fragmentation
        ### Ask for memory only when you needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        ### Limit the consumption of memory use to 2GB
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )

    def terminate(self):
        self._running = False

    def data_generator(self):
        data_path = '../processed_dataset/seperate_sign_rotate'
        c = data_reader(data_path, self.time_window)
        self.x_train, self.x_test, label_train, label_test, self.gestures = c.run()
        self.y_train = utils.one_hot(label_train, len(self.gestures))
        self.y_test = utils.one_hot(label_test, len(self.gestures))
        self.n_classes = len(self.gestures)
        print('x_train:', np.shape(self.x_train))
        print('x_test:', np.shape(self.x_test))
        print('y_train:', np.shape(self.y_train))
        print('y_test:', np.shape(self.y_test))

    def load_model(self):
        model = load_model(self.model_path)
        model.summary()
        return model

    def run(self):
        self.tf_gpus_options()
        self.data_generator()
        ##################### Load model and get weights #########################
        model = self.load_model()

        x = np.expand_dims(self.x_test[0], axis=0)
        print(x.shape)
        y_pred = model.predict(x)
        print(y_pred)
        scores = metrics.accuracy_score(self.y_test[0].argmax(1), y_pred.argmax(1), normalize=True, sample_weight=None)
        print('Test accuracy_with_thr:', scores)

        labels = self.gestures
        # Plot non-normalized confusion matrix
        save_name = self.model_path.split('/')[-1].split('_')[-2]
        save_folder = os.path.join('cm_save', self.method)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_name)
        _, cm = plot_confusion_matrix(save_path, self.y_test.argmax(1), y_pred.argmax(1),
                                      classes=labels, normalize=False,
                                      title='Confusion matrix, without normalization')
        _, cm = plot_confusion_matrix(save_path + '_normalized', self.y_test.argmax(1), y_pred.argmax(1), classes=labels, normalize=True,
                                      title='Confusion matrix, with normalization')

        ###########################################################################################


if __name__ == "__main__":
    c = Inference()
    t = Thread(target=c.run)
    t.start()
    time.sleep(10)
    # c.terminate() # Signal termination
    # t.join()





