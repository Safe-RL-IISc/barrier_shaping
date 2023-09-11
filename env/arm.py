# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi

from gym import spaces

from .base.vec_env import VecEnv
from common.torch_jit_utils import *


class Arm(VecEnv):
    def __init__(self, cfg):
        self.max_push_effort = cfg["maxEffort"]
        self.max_episode_length = cfg["max_episode_length"]

        self.spacing = cfg["envSpacing"]
        self.up_axis = cfg["sim"]["up_axis"]

        self.num_obs = 8
        self.num_act = 2

        self.use_bf = cfg["use_bf"]
        self.A = cfg["exp_A"]
        self.B = cfg["exp_B"]

        print(self.A, self.B)

        super().__init__(cfg=cfg)

        self.device = self.sim_device

        self.obs_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf
        )
        self.act_space = spaces.Box(
            np.ones(self.num_act) * -1.0, np.ones(self.num_act) * 1.0
        )

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_pos[:, 0:2] = 1
        self.goal_pos[:, 2] = 1
        # initialise envs and state tensors
        self.create_envs()

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.prepare_sim(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_tensor)
        self.rb_pos = self.rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.rb_rot = self.rb_states[:, 3:7].view(self.num_envs, self.num_bodies, 4)
        # self.rb_lvels = self.rb_states[:, 7:10].view(self.num_envs, self.num_bodies, 3)
        # self.rb_avels = self.rb_states[:, 10:13].view(self.num_envs, self.num_bodies, 3)

        self.actions = torch.zeros(
            (self.num_envs, self.num_act), dtype=torch.float32, device=self.device
        )

        self.reset()

    def create_envs(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = (
            gymapi.Vec3(0.0, 0.0, 1.0)
            if self.up_axis == "z"
            else gymapi.Vec3(0.0, 1.0, 0.0)
        )
        self.gym.add_ground(self.sim, plane_params)

        print(f"num envs {self.num_envs} env spacing {self.spacing}")

        spacing = self.spacing
        num_per_row = int(np.sqrt(self.num_envs))

        lower = (
            gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
            if self.up_axis == "z"
            else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        )
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../assets"
        )
        asset_file = "urdf/arm.urdf"

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(
        #         os.path.dirname(os.path.abspath(__file__)),
        #         self.cfg["env"]["asset"].get("assetRoot", asset_root),
        #     )
        #     asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(cartpole_asset)

        pose = gymapi.Transform()
        if self.up_axis == "z":
            pose.p.x = 0.0
            pose.p.y = 0.0
            pose.p.z = 0.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2)

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            cartpole_handle = self.gym.create_actor(
                env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0
            )

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props["driveMode"][0] = gymapi.DOF_MODE_EFFORT
            dof_props["driveMode"][1] = gymapi.DOF_MODE_EFFORT
            dof_props["stiffness"][:] = 0.0
            dof_props["damping"][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        _, _, yaw = get_euler_xyz(self.rb_rot[env_ids, 2, :])
        _, _, yaw2 = get_euler_xyz(self.rb_rot[env_ids, 1, :])

        # maps rpy from -pi to pi
        yaw = torch.where(yaw > torch.pi, yaw - 2 * torch.pi, yaw)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        ARM_LEN = 0.8

        self.obs_buf[env_ids, 4] = (
            self.rb_pos[env_ids, 2, 0].squeeze()
            - self.goal_pos[env_ids, 0]
            + ARM_LEN * torch.sin(yaw[env_ids])
        )
        self.obs_buf[env_ids, 5] = (
            self.rb_pos[env_ids, 2, 1].squeeze()
            - self.goal_pos[env_ids, 1]
            - ARM_LEN * torch.cos(yaw[env_ids])
        )

        self.obs_buf[env_ids, 6] = yaw2[env_ids]
        self.obs_buf[env_ids, 7] = yaw[env_ids]

        return self.obs_buf

    def get_reward(self):
        x = self.obs_buf[:, 4]
        y = self.obs_buf[:, 5]
        a1 = self.obs_buf[:, 0]
        a2 = self.obs_buf[:, 2]

        l = torch.tensor([-3.14, -3], device="cuda:0")
        h = torch.tensor([3.14, 3], device="cuda:0")

        # pole_angle = self.obs_buf[:, 2]
        # pole_vel = self.obs_buf[:, 3]
        # cart_vel = self.obs_buf[:, 1]
        # cart_pos = self.obs_buf[:, 0]

        (
            self.reward_buf[:],
            self.reset_buf[:],
            self.return_buf[:],
        ) = compute_cartpole_reward(
            x,
            y,
            a1,
            a2,
            l,
            h,
            self.A,
            self.B,
            self.dof_pos,
            self.dof_vel,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
            self.return_buf,
        )

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        positions = 0.2 * (
            torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5
        )
        velocities = 0.5 * (
            torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5
        )

        goaly = 1.25 * (torch.rand(len(env_ids), device=self.device) - 1)
        goalx = 1.25 * 2 * (torch.rand(len(env_ids), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]
        self.goal_pos[env_ids, 0] = goalx[:]
        self.goal_pos[env_ids, 1] = goaly[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.return_buf[env_ids] = 0

        self.get_obs()

    def step(self, actions):
        actions_tensor = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=torch.float
        )
        # actions_tensor[:: self.num_dof] = (
        #     actions.to(self.device).squeeze() * self.max_push_effort
        # )

        actions_tensor[:, 0] = actions[:, 0].to(self.device)
        actions_tensor[:, 1] = actions[:, 1].to(self.device) * 0.01
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        # simulate and render
        self.simulate()
        if not self.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()

    def _generate_lines(self):
        num_lines = 1
        line_colors = [[0, 0, 0]]

        line_vertices = []

        for i in range(self.num_envs):
            vertices = [
                [self.goal_pos[i, 0].item(), self.goal_pos[i, 1].item(), 0],
                [
                    self.goal_pos[i, 0].item(),
                    self.goal_pos[i, 1].item(),
                    self.goal_pos[i, 2].item(),
                ],
                # [
                #     self.goal_pos[i, 0].item(),
                #     self.goal_pos[i, 1].item(),
                #     self.goal_pos[i, 2].item(),
                # ],
                # [
                #     self.goal_pos[i, 0].item() + math.cos(self.goal_rot[i, 0].item()),
                #     self.goal_pos[i, 1].item() + math.sin(self.goal_rot[i, 0].item()),
                #     self.goal_pos[i, 2].item(),
                # ],
                # [
                #     self.obs_buf[i, 4].item() + self.goal_pos[i, 0].item(),
                #     self.obs_buf[i, 5].item() + self.goal_pos[i, 1].item(),
                #     0,
                # ],
                # [
                #     self.obs_buf[i, 4].item() + self.goal_pos[i, 0].item(),
                #     self.obs_buf[i, 5].item() + self.goal_pos[i, 1].item(),
                #     1,
                # ],
            ]

            line_vertices.append(vertices)

        return line_vertices, line_colors, num_lines


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_cartpole_reward(
    x,
    y,
    a1,
    a2,
    l,
    h,
    A,
    B,
    dof_pos,
    dof_vel,
    reset_buf,
    progress_buf,
    max_episode_length,
    return_buf,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = torch.exp(-(x**2 + y**2))
    # reward = -abs(x) - abs(y)

    bf = A * (1 - (torch.exp(B * (dof_pos - h)) + torch.exp(B * (l - dof_pos)))) + \
         A * B * dof_vel * (torch.exp(B * (l - dof_pos)) - torch.exp(B * (dof_pos - h)))

    # adjust reward for reset agents
    reward = torch.where(torch.abs(a1) > 3, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(a2) > 2.9, torch.ones_like(reward) * -2.0, reward)

    return_buf += reward

    reward += torch.sum(bf, dim=1)

    reset = torch.where(torch.abs(a1) > 3.14, torch.ones_like(reset_buf), reset_buf)

    reset = torch.where(torch.abs(a2) > 3, torch.ones_like(reset_buf), reset)
    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset
    )

    return reward, reset, return_buf
