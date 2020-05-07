import os, json
import numpy as np
import matplotlib.pyplot as plt

class data_reader():
    def __init__(self, data_path, time_window):
        self.data_path = data_path
        self.time_window = time_window
        self.print_file = open('print_save.txt', 'w+')

    def time_window_set(self, samples, one_gesture, time_window):
        num_samples = len(one_gesture) - time_window
        for i in range(num_samples):
            one_sample = one_gesture[i:i+time_window]
            #self.bar_plot(one_sample)
            samples.append(one_sample)
        return samples

    def bar_plot(self, feature_normalized):
        self.fig = plt.figure()
        ax1 = self.fig.add_subplot(111)
        for i in range(len(feature_normalized)):
            x = np.linspace(0, len(feature_normalized[i]), len(feature_normalized[i]))
            y = feature_normalized[i]
            for j in range(len(feature_normalized[i])):
                ax1.bar(x[j], y[j], color='blue')
            plt.pause(1)
            plt.cla()

    def run(self):
        gestures, x_train, x_test, y_train, y_test = [], [], [], [], []
        for gesture in os.listdir(self.data_path):
            gestures.append(gesture)
            label = len(gestures)-1
            print('====================>>' + gesture + ' ' + str(label) + '<<====================')
            folder_path = os.path.join(self.data_path, gesture)
            samples_per_gesture_train = []
            samples_per_gesture_test = []

            for person in os.listdir(folder_path):
                samples_path = os.path.join(folder_path, person)
                one_gesture = np.load(samples_path)
                # 25 persons in each gesture, 18 for training and 6 for test
                person_index = int(person.split('.')[0].split('_')[0][-2:])
                test_person_index = [19, 20, 21, 22, 23, 24, 25]
                if person_index in test_person_index:
                    samples_per_gesture_test = self.time_window_set(samples_per_gesture_test, one_gesture, time_window=self.time_window)
                else:
                    samples_per_gesture_train = self.time_window_set(samples_per_gesture_train, one_gesture, time_window=self.time_window)
            print('Train', np.shape(samples_per_gesture_train))
            print('Test', np.shape(samples_per_gesture_test))
            Label_train = np.ones((len(samples_per_gesture_train), 1)) * label
            Label_test = np.ones((len(samples_per_gesture_test), 1)) * label
            print('Label train', np.shape(Label_train))
            print('Label test', np.shape(Label_test))

            ## Training and test data, label
            x_train.extend(samples_per_gesture_train)
            x_test.extend(samples_per_gesture_test)
            y_train.extend(Label_train)
            y_test.extend(Label_test)
            print('x_train', np.shape(x_train))
            print('x_test', np.shape(x_test))
            print('y_train', np.shape(y_train))
            print('y_test', np.shape(y_test))
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), gestures

if __name__ == '__main__':
    data_path = '../processed_dataset/frame_step_1'
    c = data_reader(data_path, time_window=25)
    c.run()

# boxing, handclapping, handwaving, jogging, running, walking