#coding : utf-8
import shutil
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class OpenposeLauncher():
    def __init__(self,openpose_path, video_path, jason_output_path, is_display, render_pose, frame_step):
        '''
        :param dir_contains_models:  the dir that have dir "models", e.g. "G:/openpose"
        :param openpose_binary_path: binary file of openpose e.g. "G:/openpose/bin/OpenPoseDemo.exe"
        '''
        self.video_folder = video_path
        self.jason_output_folder = jason_output_path
        self.openpose_path = openpose_path
        self.display = is_display
        self.render_pose = render_pose  # --display 0 --render_pose 0
        self.frame_step = frame_step

    def openpose_image(self,image_dir,log_output_dir):
        os.chdir(self.openpose_path)
        print(os.getcwd())
        command = "\"%s\" --image_dir=%s --write_json=%s --logging_level 3 " \
                  % (self.openpose_path, image_dir, log_output_dir)
        os.system(command)

    def openpose_camera(self,log_output_dir,camera_index=0):
        os.chdir(self.model_dir)
        # command = "\"%s\" --write_json %s --model_pose COCO --number_people_max 1 --camera %s --logging_level 3 " % (self.openpose_path, log_output_dir, camera_index)
        command = "\"%s\" --write_json %s --model_pose BODY_25 -num_gpu 1 --number_people_max -1 --camera %s --logging_level 3 " % (
        self.openpose_path, log_output_dir, camera_index)
        os.system(command)

    def openpose_video(self):
        print(os.getcwd())
        if os.getcwd() != 'F:\动作识别_KTH\openpose-1.6.0-binaries-win64-gpu-python-flir-3d_recommended\openpose':
            # 前往当前工作目录 含有models的主文件夹
            os.chdir(self.openpose_path)
        print(os.getcwd())
        os.makedirs(self.jason_output_folder, exist_ok=True)
        if self.display == 0:
            command = "\"%s\" --model_pose BODY_25 --video=%s --number_people_max -1 --write_json=%s --part_candidates 1 --display %s --render_pose %s --frame_step %s"\
                      % ('bin\OpenPoseDemo.exe', self.video_folder, self.jason_output_folder, self.display, self.render_pose, self.frame_step)
        elif self.display == 1:
            command = "\"%s\" --model_pose BODY_25 --video=%s --number_people_max -1 --write_json=%s --part_candidates 1 --frame_step %s"\
                      % ('bin\OpenPoseDemo.exe', self.video_folder, self.jason_output_folder, self.frame_step)
        os.system(command)

    def openpose_IP_camera(self,camera_ip,log_output_dir):
        '''
    get public ip camera: http://www.webcamxp.com/publicipcams.aspx

    '''
        os.chdir(self.model_dir)
        raise NotImplementedError()

    def openpose_hands(self):
        pass

if __name__ == '__main__':
    # ../KTH_data/boxing/person01_boxing_d1_uncomp.avi
    video_path = '../../KTH_data/boxing/person01_boxing_d1_uncomp.avi'
    jason_output_path = '../../Feature_extraction/joint_outputs'
    openpose_path = '../openpose-1.6.0-binaries-win64-gpu-python-flir-3d_recommended/openpose'
    os.makedirs(jason_output_path, exist_ok=True)
    c = OpenposeLauncher(openpose_path, video_path, jason_output_path, is_display=1, render_pose=1, frame_step=5)
    c.openpose_video()
