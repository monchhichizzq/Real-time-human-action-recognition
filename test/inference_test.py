import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    def __init__(self, model_path, x_list, gestures):
        self.x_list = x_list
        self.model_path = model_path
        self.gestures = gestures

    def load_model(self):
        model = load_model(self.model_path)
        #model.summary()
        return model

    def terminate(self):
        self._running = False

    def run(self):
        # print('model', os.getcwd())
        # self.tf_gpus_options()
        # self.set_config()
        model = self.load_model()

        if np.shape(self.x_list)[0]== 0:
            pass
        else:
            #x_test = np.expand_dims(self.x_list,axis=0)
            x_test = np.array(self.x_list)
            # print(np.shape(x_test))
            ##################### Load model and get weights #########################
            y_pred = model.predict(x_test)
            labels = self.gestures
            print('predictions', y_pred.shape)
            # print(y_pred.argmax(1))
            y_pred_label = [labels[i] for i in y_pred.argmax(1)]
            print(y_pred_label)
            # print(labels[int(y_pred.argmax())])


if __name__ == "__main__":
    model_path = 'models/HG-KTH_model_-0.91_.h5'
    gestures = ['boxing', 'handwaving', 'handclapping', 'jogging', 'running', 'walking']
    x_list = np.ones((20,14))
    c = Inference(model_path, x_list, gestures)
    t = Thread(target=c.run)
    t.start()
    # time.sleep(10)
    # # c.terminate() # Signal termination
    # # t.join()





