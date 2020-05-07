import os
import time
import shutil
import threading
import _thread
import numpy as np
from collections import Counter
from Read_videos import Video_reader
from threading import Thread
from jason_2_input import jason_reader
from keras.models import load_model
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from plot_animation import bar_plot


class Inference():
    def __init__(self, model, x_list, gestures):
        self.x_list = x_list
        self.model = model
        self.gestures = gestures

    # def load_model(self):
    #     model = load_model(self.model_path)
    #     #model.summary()
    #     return model

    def run(self):
        model = self.model

        if np.shape(self.x_list)[0]== 0:
            pass
        else:
            #x_test = np.expand_dims(self.x_list,axis=0)
            x_test = np.array(self.x_list)
            # print(np.shape(x_test))
            ##################### Load model and get weights #########################
            y_pred = model.predict(x_test)
            labels = self.gestures
            #print('predictions', y_pred.shape)
            # print(y_pred.argmax(1))
            y_pred_label = [labels[i] for i in y_pred.argmax(1)]
            print(y_pred_label)


class Motion_detector():
    def __init__(self):
        self.gpu_lstm = 128
        self.gpu_openpose = 4096
        self.gpu0_free = 512
        self.time_step = 20
        self.frame_step = 1
        self.video_path = 'video_samples'
        self.jason_output_path = '../test/outputs_' + str(self.frame_step)
        #self.model_path = '../../test/models/HG-KTH_model_-0.91_.h5'
        self.model_path =  'models/HG-KTH_model_-0.91_.h5'
        self.predictions = ['Waiting']
        self.gestures = ['boxing', 'clapping', 'waving', 'jogging', 'running', 'walking', 'No one', 'Waiting', ' ']
        # self.predictions = ['请稍等。。。']
        # self.gestures = ['打拳', '拍手', '挥手', '慢跑', '跑步', '走路', '没有人', '请稍等。。。', ' ']
        self.gestures_container = list(np.ones((100, self.time_step, 14)))


    def tf_gpus_options(self):
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        print(gpus, cpus)
        ### By default, TensorFlow requests and blocks nearly all of the GPU memory of the two available GPUs to avoid memory fragmentation
        ### Ask for memory only when you needed
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        ### Limit the consumption of memory use to 2GB
        ### Create two virtual gpu in one GPU
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit= self.gpu_lstm),
                                                                tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit= self.gpu0_free),
                                                                 tf.config.experimental.VirtualDeviceConfiguration(
                                                                     memory_limit= self.gpu_openpose )])
        # print(tf.config.get_soft_device_placement)
        print(tf.config.experimental_list_devices())

    def load_model(self):
        global model
        model = load_model(self.model_path)
        model.summary()
        model._make_predict_function()
        # print('test', model.predict(np.ones((1,20,14))))


    def renew_folder(self, folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(self.jason_output_path, exist_ok=True)
        else:
            os.makedirs(self.jason_output_path, exist_ok=True)

    def pred(self):
        # print(np.shape(self.gestures_container))
        #predictor = Inference(self.model, self.gestures_container, self.gestures)
        #thread_p = Thread(target=predictor.run)
        thread_p = Thread(target=self.model_run)
        thread_p.setName('predictor')
        #thread_p.daemon = 1
        thread_p.start()

    def accumulate_input(self):
        with tf.device('/gpu:0'):
            self.load_model()
            while True:
                self.model_run()
                self.gestures_container = []
                time.sleep(0.001)

    def call(self):
        j_reader = jason_reader('../' + self.jason_output_path, self.time_step)
        fig = plt.figure(figsize=(8, 6))
        count_no_person = 0
        while True:
            self.gesture_list = j_reader.run()
            if len(self.gesture_list) == 1:
                count_no_person+=1
                if count_no_person >= 40:
                    # print('No one')
                    self.predictions.append("No one")
                    # self.predictions.append("没有人")

                    count_no_person = 0
            else:
                if np.shape(self.gesture_list)[0] == 20:
                    # print(np.shape(self.gesture_list))
                    self.gestures_container.append(self.gesture_list)
            #print('Predictions:', self.predictions[-1], self.gestures.index(self.predictions[-1]))
            bar_plot(fig, self.gestures, self.predictions[-1])
            time.sleep(0.01)

    def model_run(self):
        input = self.gestures_container
        if np.shape(input)[0]== 0:
            pass
        else:
            #x_test = np.expand_dims(self.x_list,axis=0)
            x_test = np.array(input)
            ##################### Load model and get weights #########################
            y_pred = model.predict(x_test)
            labels = self.gestures
            y_pred_label = [labels[i] for i in y_pred.argmax(1)]
            gesture_most = Counter(y_pred_label).most_common(1)[0][0]
            gesture_most_label = self.gestures.index(gesture_most)
            if x_test.shape[0] != 100:
                self.predictions.append(gesture_most)
                #self.prob

    def run(self):
        self.tf_gpus_options()
        self.renew_folder(self.jason_output_path)

        thread_pred = Thread(target=self.accumulate_input)
        thread_pred.setName('Accumulation_preds')
        #thread_jr.daemon = 1
        thread_pred.start()

        thread_jr = Thread(target=self.call)
        thread_jr.setName('Jason_Reader')
        # thread_jr.daemon = 1
        thread_jr.start()

        print('Threads number:', threading.activeCount())
        print(threading.enumerate())


        # # # Run openpose and get the jason outputs
        # with tf.device('/gpu:3'):
        #     skeleton_extractor = Video_reader(self.video_path, self.jason_output_path, self.frame_step)
        #     thread_se = Thread(target=skeleton_extractor.run())
        #     thread_se.setName('skeleton_extractor')
        #     thread_se.daemon = 1
        #     thread_se.start()


        ### Wait 5 seconds for the model to warm up
        with tf.device('/gpu:3'):
            skeleton_extractor = Video_reader(self.video_path, self.jason_output_path, self.frame_step)
            thread_se = threading.Timer(10, skeleton_extractor.run)
            thread_se.setName('skeleton_extractor')
            thread_se.daemon = 1
            thread_se.start()

        self.run_call = False



if __name__ == '__main__':
    detector = Motion_detector()
    thread = Thread(target=detector.run())
    thread.setName('Main')
    thread.daemon = 1
    thread.start()
    time.sleep(200)

    # j_reader = jason_reader('../' + self.jason_output_path, time_step=20)
    # thread_jr = threading.Timer(1, j_reader.read_folder)
    # thread_jr.setName('Jason_Reader')
    # thread_jr.daemon = 1
    # thread_jr.start()
    #
    # for thread in threading.enumerate():
    #     thread_name = thread.getName()
    #     print(thread_name)

