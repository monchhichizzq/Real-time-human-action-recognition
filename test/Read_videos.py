import os
from cmd_openpose import OpenposeLauncher

class Video_reader():
    def __init__(self, video_path, jason_output_path, frame_step):
        self.video_path = video_path
        self.jason_output_path = jason_output_path
        self.frame_step = frame_step

    def feature_extractor(self, video_path, output_path):
        openpose_path = '../openpose-1.6.0-binaries-win64-gpu-python-flir-3d_recommended/openpose'
        gesture_video_path = '../../test/' + video_path
        print('Openpose input video:', gesture_video_path)
        output_path = '../' + output_path
        print('Openpose output jasons:', output_path )
        extractor = OpenposeLauncher(openpose_path, gesture_video_path, output_path, is_display=1, render_pose=1, frame_step=self.frame_step)
        extractor.openpose_video()

    def run(self):
        videos_path = self.video_path
        if os.getcwd() == 'F:\动作识别_KTH\openpose-1.6.0-binaries-win64-gpu-python-flir-3d_recommended\openpose':
            os.chdir('../../test')
        for video in os.listdir(videos_path):
            if video.endswith('.avi'):
                print('Reading' + video)
                gesture_video_path = os.path.join(videos_path, video)
                output_path = self.jason_output_path
                self.feature_extractor(gesture_video_path, output_path)

if __name__ == '__main__':
    frame_step = 1
    video_path = 'video_samples'
    jason_output_path = '../test/outputs_' + str(frame_step)
    os.makedirs(jason_output_path, exist_ok=True)
    c = Video_reader(video_path, jason_output_path, frame_step)
    c.run()