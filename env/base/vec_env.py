from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gym import spaces

from common.torch_jit_utils import *
import sys

import torch
import math


class VecEnv:
    def __init__(self, cfg):
        self.compute_device_id = cfg["sim"].get("compute_device_id", 0)
        self.graphics_device_id = cfg["sim"].get("graphics_device_id", 0)
        self.sim_device = cfg["sim"].get("sim_device", "cuda:0")
        self.headless = cfg["sim"].get("headless", False)

        self.num_envs = cfg.get("num_envs", 512)
        self.max_episode_length = cfg.get("max_episode_length", 500)

        self.rand_vel_targets = cfg["task"].get("rand_vel_targets", False)
        self.train = cfg.get("mode", "train")

        self.goal_lim = cfg.get("goal_lim", 10)
        self.vel_lim = cfg.get("vel_lim", 5)
        self.goal_vel_lim = cfg.get("goal_vel_lim", 3)

        # configure sim (gravity is pointing down)
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        grav = cfg["sim"].get("gravity", (0.0, 0.0, 0.0))
        self.sim_params.gravity = gymapi.Vec3(grav[0], grav[1], grav[2])
        self.sim_params.dt = cfg["sim"].get("dt", 1 / 60)
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = True

        # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
        
        cfgPhy = cfg["sim"].get("physx", {})
        self.sim_params.physx.num_position_iterations = cfgPhy.get("num_position_iterations", 4)
        self.sim_params.physx.num_velocity_iterations = cfgPhy.get("num_velocity_iterations", 0)
        self.sim_params.physx.rest_offset = cfgPhy.get("rest_offset", 0.001)
        self.sim_params.physx.contact_offset = cfgPhy.get("contact_offset", 0.02)
        self.sim_params.physx.use_gpu = cfgPhy.get("use_gpu ", True)
        self.sim_params.physx.num_threads = cfgPhy.get("num_threads", 4)
        self.sim_params.physx.solver_type = cfgPhy.get("solver_type", 1)
        self.sim_params.physx.bounce_threshold_velocity = cfgPhy.get("bounce_threshold_velocity", 0.2)
        self.sim_params.physx.max_depenetration_velocity = cfgPhy.get("max_depenetration_velocity", 10.0)
        self.sim_params.physx.default_buffer_size_multiplier = cfgPhy.get("default_buffer_size_multiplier",5.0)
        self.sim_params.physx.max_gpu_contact_pairs = cfgPhy.get("max_gpu_contact_pairs",8388608)
        self.sim_params.physx.num_subscenes = cfgPhy.get("num_subscenes",4)
        #self.sim_params.physx.contact_collection = cfgPhy.get("contact_collection", 0)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(
            self.compute_device_id,
            self.graphics_device_id,
            gymapi.SIM_PHYSX,
            self.sim_params,
        )
        

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.sim_device
        )
        self.reward_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self.return_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self.truncated_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.sim_device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.sim_device, dtype=torch.long
        )
        self.obs_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf
        )
        self.act_space = spaces.Box(
            np.ones(self.num_act) * -1.0, np.ones(self.num_act) * 1.0
        )
        # generate viewer for visualisation
        self.set_viewer()

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

            # set the camera position based on up axis
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_envs(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_reward(self):
        pass

    def reset(self):
        raise NotImplementedError

    def simulate(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def _generate_lines(self):
        line_vertices = None
        line_colors = None
        num_lines = 0
        return line_vertices, line_colors, num_lines

    def render(self):
        if self.viewer:
            # update viewer
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)

                line_vertices, line_colors, num_lines = self._generate_lines()

                if num_lines:
                    for i, envi in enumerate(self.envs):
                        self.gym.add_lines(
                            self.viewer, envi, num_lines, line_vertices[i], line_colors
                        )

                self.gym.draw_viewer(self.viewer, self.sim, True)

                self.gym.clear_lines(self.viewer)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

    def exit(self):
        # close the simulator in a graceful way
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
