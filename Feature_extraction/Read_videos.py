import os
from cmd_openpose import OpenposeLauncher

class Video_reader():
    def __init__(self, video_path, jason_output_path, frame_step):
        self.video_path = video_path
        self.jason_output_path = jason_output_path
        self.frame_step = frame_step

    def feature_extractor(self, video_path, output_path):
        openpose_path = '../openpose-1.6.0-binaries-win64-gpu-python-flir-3d_recommended/openpose'
        gesture_video_path = '../' + video_path
        output_path = '../' + output_path
        extractor = OpenposeLauncher(openpose_path, gesture_video_path, output_path, is_display=1, render_pose=1, frame_step=self.frame_step)
        extractor.openpose_video()

    def run(self):
        gestures = ['None']
        for gesture in os.listdir(self.video_path):
            if not gesture.endswith('.zip'):
                gestures.append(gesture)
                print('====================>>'+ gesture +'<<====================')
                videos_path = os.path.join(self.video_path, gesture)
                print(os.getcwd())
                if os.getcwd() == 'F:\动作识别_KTH\openpose-1.6.0-binaries-win64-gpu-python-flir-3d_recommended\openpose':
                    os.chdir('../../Feature_extraction')
                for video in os.listdir(videos_path):
                    if video.endswith('.avi'):
                        print('Reading' + video)
                        video_index = video.split('_')[0] + '_'+ video.split('_')[-2]
                        print(video_index)
                        gesture_video_path = os.path.join(videos_path, video)
                        output_path = os.path.join(self.jason_output_path, gesture, video_index)
                        self.feature_extractor(gesture_video_path, output_path)
        print(gestures)

if __name__ == '__main__':
    frame_step = 1
    video_path = '../KTH_data'
    jason_output_path = '../KTH_openpose_outputs_' + str(frame_step)
    os.makedirs(jason_output_path, exist_ok=True)
    c = Video_reader(video_path, jason_output_path, frame_step)
    c.run()