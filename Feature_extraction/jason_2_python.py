import os, json
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2

class jason_reader():
    def __init__(self, jason_path):
        self.jason_path = jason_path

    def neck_center(self, neck, data_part_candidate):
        # x, y, confidence
        if len(data_part_candidate) != 0 and len(neck) != 0:
            centerlized = [data_part_candidate[0] - neck[0], data_part_candidate[1] - neck[1]]
            return centerlized
        else:
            return data_part_candidate

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def joints_plot(self, data_part_candidates):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(25):
            print('coordinate', i, data_part_candidates[str(i)])
            if len(data_part_candidates[str(i)])!=0:
                print(data_part_candidates[str(i)][0], data_part_candidates[str(i)][1])
                ax.scatter(-data_part_candidates[str(i)][0], -data_part_candidates[str(i)][1], marker='o', color='blue')
        plt.pause(0.1)

    def joints_visualization(self, data_part_candidates):
        frame = np.ones((160,120))*255
        for i in range(25):
            if len(data_part_candidates[str(i)]) != 0:
                print(data_part_candidates[str(i)][0], data_part_candidates[str(i)][1], frame.shape)
                frame[int(data_part_candidates[str(i)][0]), int(data_part_candidates[str(i)][1])] = 0
        img90 = self.rotate(frame, -90)
        cv2.imshow('run', img90)
        cv2.waitKey(1)

    def joints_visualization_after_centered(self, joints_centered):
        frame = np.ones((360, 120)) * 255
        for i in range(25):
            if len(joints_centered[i]) != 0 and len(joints_centered[1]) != 0:
                frame[int(joints_centered[i][0]+180), int(joints_centered[i][1])+20] = 0
        img90 = self.rotate(frame, -90)
        cv2.imshow('run', img90)
        cv2.waitKey(1)

    def bar_plot(self, feature_normalized):
        ax1 = self.fig.add_subplot(111)
        x = np.linspace(0, len(feature_normalized), len(feature_normalized))
        y = feature_normalized
        for i in range(len(feature_normalized)):
            ax1.bar(x[i], y[i], color='blue')
        plt.pause(1)
        plt.cla()

    def abs_distance(self, joint_0, joint_1):
        d = np.sqrt((joint_1[0] - joint_0[0]) ** 2 + (joint_1[1] - joint_0[1]) ** 2)
        return d

    def normalized(self, dis, dis_normal):
        dis_new = dis/dis_normal
        return dis_new

    def time_seq_feature(self, joints_centered, bar_plot):
        # 0-1 neck
        # 1-8 body
        # 1-2, 1-3, 1-4 left hand
        # 1-5, 1-6, 1-7 right hand
        # 8-10, 8-11 left leg
        # 8-13, 8-14 right leg
        # 3-6, 4-7 hand distance
        # 10-13, 11-16 leg distance

        # Body
        dis_1_8 = self.abs_distance(joints_centered[1], joints_centered[8])

        # Left hand
        dis_1_2 = self.abs_distance(joints_centered[1], joints_centered[2])
        dis_1_3 = self.abs_distance(joints_centered[1], joints_centered[3])
        dis_1_4 = self.abs_distance(joints_centered[1], joints_centered[4])

        # Right hand
        dis_1_5 = self.abs_distance(joints_centered[1], joints_centered[5])
        dis_1_6 = self.abs_distance(joints_centered[1], joints_centered[6])
        dis_1_7 = self.abs_distance(joints_centered[1], joints_centered[7])

        # Left leg
        dis_8_10 = self.abs_distance(joints_centered[8], joints_centered[10])
        dis_8_11 = self.abs_distance(joints_centered[8], joints_centered[11])

        # Right leg
        dis_8_13 = self.abs_distance(joints_centered[8], joints_centered[13])
        dis_8_14 = self.abs_distance(joints_centered[8], joints_centered[14])

        # hand distance
        dis_3_6 = self.abs_distance(joints_centered[3], joints_centered[6])
        dis_4_7 = self.abs_distance(joints_centered[4], joints_centered[7])

        # leg distance
        dis_10_13 = self.abs_distance(joints_centered[10], joints_centered[13])
        dis_11_14 = self.abs_distance(joints_centered[11], joints_centered[14])

        # 3+3+2+2
        feature = [dis_1_2, dis_1_3, dis_1_4, dis_1_5, dis_1_6, dis_1_7, dis_8_10, dis_8_11, dis_8_13, dis_8_14]
        feature = [dis_3_6, dis_4_7, dis_1_2, dis_1_3, dis_1_4, dis_1_5, dis_1_6, dis_1_7, dis_10_13, dis_11_14, dis_8_10, dis_8_11, dis_8_13, dis_8_14]

        # Normalized
        feature_normlized = [self.normalized(dis, dis_1_8) for dis in feature]
        if bar_plot:
            self.bar_plot(feature_normlized/max(feature_normlized))
        #print('feature', feature)
        #print('New features', feature_normlized)
        #print('Normalized features', feature_normlized/max(feature_normlized))
        return feature_normlized/max(feature_normlized)

    def no_empty_in_the_list(self, joints_centered):
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
        main_joints_15_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        main_joints_15 = [joints_centered[i] for i in main_joints_15_index]
        if [] in main_joints_15:
            # print(main_joints_15)
            # print('There is empty in the list')
            return False
        else:
            return True

    def run(self):
        gestures = ['None']
        for gesture in os.listdir(self.jason_path):
            gestures.append(gesture)
            print('====================>>' + gesture + '<<====================')
            folder_path = os.path.join(self.jason_path, gesture)
            for person in os.listdir(folder_path):
                print('Reading' + person)
                samples_path = os.path.join(folder_path, person)
                gesture_per_person = []
                self.fig = plt.figure()
                for js in os.listdir(samples_path):
                    with open(os.path.join(samples_path, js)) as json_file:
                        # print('====================>>' + os.path.join(samples_path, js) + '<<====================')
                        data = json.load(json_file)
                        #print('Orignal', data['part_candidates'])
                        data_part_candidates = data['part_candidates'][0]
                        neck = data_part_candidates['1']                       # Neck 1
                        bot_center = data_part_candidates['8']                 # bot_center 8
                        joints_centered = [self.neck_center(neck, data_part_candidates[str(i)]) for i in range(25)]
                        # len(np.shape(joints_centered)) 1(body detection), 2(空集, no one)
                        no_empty_in_the_list = self.no_empty_in_the_list(joints_centered)
                        if len(np.shape(joints_centered))==1 and no_empty_in_the_list:
                            self.joints_visualization_after_centered(joints_centered)
                            # self.joints_visualization(data_part_candidates)
                            # if len(neck) != 1 and len(bot_center) != 1:
                            normalized_feature = self.time_seq_feature(joints_centered, bar_plot =False)
                            gesture_per_person.append(normalized_feature)
                print(gesture, person, np.shape(gesture_per_person))
                plt.close('all')

if __name__ == '__main__':
    jason_path = '../KTH_openpose_outputs_1_less'
    c = jason_reader(jason_path)
    c.run()




########################################################################################################################
# Result for BODY_25 (25 body parts consisting of COCO + foot)
# const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
# {0,  "Nose"},
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "MidHip"},
# {9,  "RHip"},
# {10, "RKnee"},
# {11, "RAnkle"},
# {12, "LHip"},
# {13, "LKnee"},
# {14, "LAnkle"},
# {15, "REye"},
# {16, "LEye"},
# {17, "REar"},
# {18, "LEar"},
# {19, "LBigToe"},
# {20, "LSmallToe"},
# {21, "LHeel"},
# {22, "RBigToe"},
# {23, "RSmallToe"},
# {24, "RHeel"},
# {25, "Background"}
# };