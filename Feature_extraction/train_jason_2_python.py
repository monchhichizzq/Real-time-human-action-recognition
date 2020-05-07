import os, json
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
from collections import Counter

class jason_reader():
    def __init__(self, jason_path, output_npy):
        self.jason_path = jason_path
        self.output_npy = output_npy

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
        img = cv2.resize(img90, dsize=(800,600))
        cv2.imshow('run', img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    def joints_visualization_after_centered(self, joints_centered, gesture, person, js, save_img_path):

        frame = np.ones((360, 180)) * 255
        for i in range(15):
            if len(joints_centered[i]) != 0 and len(joints_centered[1]) != 0:
                frame[int(joints_centered[i][0]+179):int(joints_centered[i][0]+181), int(joints_centered[i][1])+39:int(joints_centered[i][1])+41] = 0
        img90 = self.rotate(frame, -90)
        #cv2.imshow(js, img90)
        img = cv2.resize(img90, dsize=(500, 500))
        cv2.putText(img, str(self.num_empty), (100, 100),  cv2.FONT_HERSHEY_SIMPLEX , 1, color=(255,0,0), thickness=2, lineType =cv2.LINE_AA)
        path = os.path.join(save_img_path, gesture, person)
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(path + '/'+js + '.png', img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    def bar_plot(self, feature_normalized):
        ax1 = self.fig.add_subplot(111)
        x = np.linspace(0, len(feature_normalized), len(feature_normalized))
        y = feature_normalized
        for i in range(len(feature_normalized)):
            ax1.bar(x[i], y[i], color='blue')
        plt.pause(0.1)
        plt.cla()

    def abs_distance(self, joint_0, joint_1):
        d = np.sqrt((joint_1[0] - joint_0[0]) ** 2 + (joint_1[1] - joint_0[1]) ** 2)
        return d

    def normalized(self, dis, dis_normal):
        dis_new = dis/dis_normal
        return dis_new

    def sign(self, joint_i, joint_1):
        if abs(joint_i[0] - joint_1[0]) != 0:
            sign = (joint_i[0] - joint_1[0])/abs(joint_i[0] - joint_1[0])
            return sign
        else:
            return 1

    def dis_foot_with_sign(self, joint_i, joint_j):
        dis_foot_with_sign = [(joint_i[0] - joint_j[0]), (joint_i[1] - joint_j[1])]
        return dis_foot_with_sign

    def angle(self, joint_x, joint_y, joint_z):
        dis_foot_with_sign_xy = np.array(self.dis_foot_with_sign(joint_y, joint_x))
        dis_foot_with_sign_yz = np.array(self.dis_foot_with_sign(joint_y, joint_z))
        dotxyyz = dis_foot_with_sign_xy.dot(dis_foot_with_sign_yz)
        twoxyyz = (np.sqrt(dis_foot_with_sign_xy.dot(dis_foot_with_sign_xy)) * np.sqrt(dis_foot_with_sign_yz.dot(dis_foot_with_sign_yz)))
        cos_value = float(dotxyyz/twoxyyz)
        angle_normalized = np.arccos(cos_value)/ np.pi
        # print('dot', dotxyyz,' two', twoxyyz ,cos_value)
        return angle_normalized

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
        sign_4 = self.sign(joints_centered[4], joints_centered[1])

        # Right hand
        dis_1_5 = self.abs_distance(joints_centered[1], joints_centered[5])
        dis_1_6 = self.abs_distance(joints_centered[1], joints_centered[6])
        dis_1_7 = self.abs_distance(joints_centered[1], joints_centered[7])
        sign_7 = self.sign(joints_centered[7], joints_centered[1])

        # Left leg
        # dis_8_10 = self.abs_distance(joints_centered[8], joints_centered[10])
        # dis_8_11 = self.abs_distance(joints_centered[8], joints_centered[11])
        # sign_10 = self.sign(joints_centered[10], joints_centered[8])
        # sign_11 = self.sign(joints_centered[11], joints_centered[8])
        dis_8_10 = self.dis_foot_with_sign(joints_centered[8], joints_centered[10])[0]
        dis_8_11 = self.dis_foot_with_sign(joints_centered[8], joints_centered[11])[0]

        # Right leg
        # dis_8_13 = self.abs_distance(joints_centered[8], joints_centered[13])
        # dis_8_14 = self.abs_distance(joints_centered[8], joints_centered[14])
        # sign_13 = self.sign(joints_centered[13], joints_centered[8])
        # sign_14 = self.sign(joints_centered[14], joints_centered[8])
        dis_8_13 = self.dis_foot_with_sign(joints_centered[8], joints_centered[13])[0]
        dis_8_14 = self.dis_foot_with_sign(joints_centered[8], joints_centered[14])[0]

        # hand distance
        dis_3_6 = self.abs_distance(joints_centered[3], joints_centered[6])
        dis_4_7 = self.abs_distance(joints_centered[4], joints_centered[7])

        # leg distance
        dis_10_13 = self.abs_distance(joints_centered[10], joints_centered[13])
        dis_11_14 = self.abs_distance(joints_centered[11], joints_centered[14])

        # Left rotation 手的摆幅
        angle_3_2_1 = self.angle(joints_centered[1], joints_centered[2], joints_centered[3])

        # Right rotation
        angle_6_5_1 = self.angle(joints_centered[1], joints_centered[5], joints_centered[6])

        # left right path 腿的摆幅
        angle_10_8_13 = self.angle(joints_centered[10], joints_centered[8], joints_centered[13])
        angle_11_8_14 = self.angle(joints_centered[11], joints_centered[8], joints_centered[14])

        # 3+3+2+2
        #feature = [dis_1_2, dis_1_3, dis_1_4, dis_1_5, dis_1_6, dis_1_7, dis_8_10, dis_8_11, dis_8_13, dis_8_14]
        #feature = [dis_3_6, dis_4_7, dis_1_3, dis_1_4, dis_1_6, dis_1_7, dis_10_13, dis_11_14, dis_8_10, dis_8_11, dis_8_13, dis_8_14]
        feature = [dis_3_6, dis_4_7, dis_1_4*sign_4, dis_1_7*sign_7, dis_10_13, dis_11_14, dis_8_10, dis_8_11, dis_8_13, dis_8_14]


        # Normalized
        feature_normlized = [self.normalized(dis, dis_1_8) for dis in feature]
        feature_normlized.extend([angle_3_2_1, angle_6_5_1, angle_10_8_13, angle_11_8_14])
        if bar_plot:
            self.bar_plot(feature_normlized/abs(max(feature_normlized)))
        #print('feature', feature)
        #print('New features', feature_normlized)
        # print('Normalized features', feature_normlized/abs(max(feature_normlized)))
        return feature_normlized/abs(max(feature_normlized))


    def nan_in_the_list(self, x):
        for item in x:
            if np.isnan(item):
                return True
            else:
                return False

    def fullfill_empty_in_the_list(self, joints_centered, gesture_reference):
        # 如果当前时刻的点未检测到用前一时刻的顶
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
        main_joints_15_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        main_joints_15 = [joints_centered[i] for i in main_joints_15_index]
        self.num_empty = self.count_empty_in_the_list(main_joints_15)
        if [] in main_joints_15:
            # print('orignal', main_joints_15)
            for i, joint in enumerate(main_joints_15):
                if len(joint) == 0:
                    main_joints_15[i] = gesture_reference[i]
            gesture_reference = main_joints_15
            #print('++++++', gesture_reference)
            return gesture_reference, False
        else:
            gesture_reference = main_joints_15
            return gesture_reference, True

    def save_files(self, gesture, person, gesture_per_person):
        os.makedirs(os.path.join(self.output_npy, gesture), exist_ok=True)
        saving_path = os.path.join(self.output_npy, gesture, person+ '.npy')
        print("Saving {}".format(saving_path))
        np.save(saving_path, gesture_per_person)

    def count_empty_in_the_list(self, list):
        len_list = [len(i) for i in list]
        c = Counter(len_list)
        num_empty = c[0]
        # print(Counter(len_list), c[0])
        return num_empty

    def run(self):
        gestures = ['None']
        for gesture in os.listdir(self.jason_path):
            gestures.append(gesture)
            print('====================>>' + gesture + '<<====================')
            folder_path = os.path.join(self.jason_path, gesture)
            gesture_reference = np.ones((15, 2))
            self.fig = plt.figure()
            if gesture == 'handclapping':
                for person in os.listdir(folder_path):
                    print('Reading' + person)
                    samples_path = os.path.join(folder_path, person)
                    gesture_per_person = []
                    # self.fig = plt.figure()
                    count = 0
                    for js in os.listdir(samples_path):
                        with open(os.path.join(samples_path, js)) as json_file:
                            # print('====================>>' + os.path.join(samples_path, js) + '<<====================')
                            data = json.load(json_file)
                            #print('Orignal', data['part_candidates'])
                            data_part_candidates = data['part_candidates'][0]
                            neck = data_part_candidates['1']                       # Neck 1
                            bot_center = data_part_candidates['8']                 # bot_center 8
                            joints_centered = [self.neck_center(neck, data_part_candidates[str(i)]) for i in range(25)]
                            # joint list 全为空
                            if len(np.shape(joints_centered)) == 2 and np.shape(joints_centered)[1]==0:
                                # print(joints_centered, len(np.shape(joints_centered)), np.shape(joints_centered))
                                # 动作append list 不为空，且动作时长大于20
                                if np.shape(gesture_per_person)[0] != 0 and  np.shape(gesture_per_person)[0]>=20:
                                    print('   ', gesture, person + '_' +str(count), np.shape(gesture_per_person))
                                    #print(person + '_' + str(count), gesture_per_person, file=print_file)
                                    #self.save_files(gesture, person + '_' +str(count), gesture_per_person)
                                    gesture_per_person = []
                                    count += 1

                            else:
                                # len(np.shape(joints_centered)) == 2 and np.shape(joints_centered)[1]==0, 空集条件
                                joints_centered_fullfilled, no_empty_in_the_list = self.fullfill_empty_in_the_list(joints_centered, gesture_reference)
                                gesture_reference = joints_centered_fullfilled
                                normalized_feature = self.time_seq_feature(joints_centered_fullfilled, bar_plot=False)
                                # normalized_feature 不存在nan 且 缺失点小于5
                                #print(count, self.nan_in_the_list(normalized_feature), self.num_empty)
                                if not self.nan_in_the_list(normalized_feature) and self.num_empty < 3:
                                    # self.bar_plot(normalized_feature)
                                    #print(joints_centered_fullfilled)
                                    #print(normalized_feature)
                                    gesture_per_person.append(normalized_feature)

                                    # save image
                                    #save_img_path = '../joints_plot_save/frame_1_seperate_no_empty'
                                    #self.joints_visualization_after_centered(joints_centered_fullfilled, gesture, person + '_' +str(count), js, save_img_path)
                    # print(gesture, person, np.shape(gesture_per_person))
                    self.save_files(gesture, person + '_' +str(count), gesture_per_person)
                    plt.close('all')


if __name__ == '__main__':
    print_file = open('print_save.txt', 'w+')
    jason_path = '../KTH_openpose_outputs_1'
    output_npy = '../processed_dataset/seperate_sign_rotate'
    c = jason_reader(jason_path, output_npy)
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