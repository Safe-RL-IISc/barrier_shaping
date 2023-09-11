import subprocess
import time


class Visualizer:
    def __init__(
        self,
        env,
        raisim_unity_path: str,
        render: bool,  # render the env
        record: bool,  # record video
        save_video_path: str,
    ) -> None:
        self.env = env
        self.raisim_unity_path = raisim_unity_path
        self.render = render
        self.record = record
        self.save_video_path = save_video_path

    def turn_on(self, episodes):
        if self.record:
            self.env.start_video_recording(
                self.save_video_path + "epi_" + str(episodes) + ".mp4"
            )

    def turn_off(self):
        if self.record:
            self.env.stop_video_recording()

    def spawn(self):
        if self.render:
            self.proc = subprocess.Popen(self.raisim_unity_path)
            self.env.turn_on_visualization()
            time.sleep(5)

    def kill(self):
        if self.render:
            self.env.turn_off_visualization()
            self.proc.kill()
