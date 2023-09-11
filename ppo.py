# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import argparse
import os
import random
import time
from distutils.util import strtobool
from env.wrapper.multiTask import MultiTaskEnv

import gym
import wandb

from env import env_map

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(env.num_obs, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(env.num_obs, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, env.num_act), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.num_act))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class PPO_agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        self.agent_cfg = cfg["agent"]
        self.buffer_cfg = cfg["buffer"]

        self.agent_cfg["batch_size"] = int(
            self.env_cfg["num_envs"] * self.agent_cfg["num_steps"]
        )
        self.agent_cfg["minibatch_size"] = int(
            self.agent_cfg["batch_size"] // self.agent_cfg["num_minibatches"]
        )

        self.device = cfg["rl_device"]

        env_spec = MultiTaskEnv(env_cfg=self.env_cfg)
        self.env, _, _ = env_spec.getEnv()

        # #### temp
        # self.low = torch.ones_like(self.env.dof_pos, device="cuda:0") * 20
        # self.high = torch.ones_like(self.env.dof_pos, device="cuda:0") * (-20)

        self.run_name = (
            f"{self.env_cfg['env_name']}__{self.agent_cfg['name']}__{int(time.time())}"
        )

        self.writer = SummaryWriter(f"runs/{self.run_name}")

        self.agent = Agent(self.env).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.agent_cfg["learning_rate"], eps=1e-5
        )

        self._init_buffers()
        ###

    def _init_buffers(self):
        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"], self.env.num_obs),
            dtype=torch.float,
        ).to(self.device)
        self.actions = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"], self.env.num_act),
            dtype=torch.float,
        ).to(self.device)
        self.logprobs = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.rewards = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.dones = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.values = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.advantages = torch.zeros_like(self.rewards, dtype=torch.float).to(
            self.device
        )

    def update(self):
        raise NotImplemented

    def run(self):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()

        # next_obs = envs.reset()
        next_obs = self.env.obs_buf

        next_done = torch.zeros(self.env_cfg["num_envs"], dtype=torch.float).to(
            self.device
        )
        num_updates = self.agent_cfg["total_timesteps"] // self.agent_cfg["batch_size"]

        for update in range(1, num_updates + 1):
            metrics = {}
            # Annealing the rate if instructed to do so.
            if self.agent_cfg["anneal_lr"]:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.agent_cfg["learning_rate"]
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.agent_cfg["num_steps"]):
                global_step += 1 * self.env_cfg["num_envs"]
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.

                # next_obs, rewards[step], next_done, info = envs.step(action)

                self.env.step(action)
                next_obs, self.rewards[step], next_done, episodeLen, episodeRet = (
                    self.env.obs_buf,
                    self.env.reward_buf,
                    self.env.reset_buf.clone(),
                    self.env.progress_buf.clone(),
                    self.env.return_buf.clone(),
                )
                self.env.reset()

                # if 0 <= step <= 2:
                done_ids = next_done.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.size()[0]:
                    # taking mean over all envs that are done at the
                    # current timestep
                    episodic_return = torch.mean(episodeRet[done_ids].float()).item()
                    episodic_length = torch.mean(episodeLen[done_ids].float()).item()
                    if step % 100 == 0:
                        print(
                            f"global_step={global_step}, episodic_return={episodic_return}"
                        )

                        # if episodic_return > 50:
                        #     dof_pos_scaled = (self.env.dof_pos - self.env.default_dof_pos) * self.env.dof_pos_scale
                        #     self.low = torch.where(self.low < dof_pos_scaled, self.low, dof_pos_scaled)
                        #     self.high = torch.where(self.high > dof_pos_scaled, self.high, dof_pos_scaled)

                    self.writer.add_scalar("rewards/step", episodic_return, global_step)
                    self.writer.add_scalar(
                        "episode_lengths/step", episodic_length, global_step
                    )

                    metrics.update({
                        "rewards/step": episodic_return,
                        "episode_lengths/step": episodic_length,
                    })

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.agent_cfg["num_steps"])):
                    if t == self.agent_cfg["num_steps"] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = (
                        self.rewards[t]
                        + self.agent_cfg["gamma"] * nextvalues * nextnonterminal
                        - self.values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.agent_cfg["gamma"]
                        * self.agent_cfg["gae_lambda"]
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1, self.env.num_obs))
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1, self.env.num_act))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(self.agent_cfg["update_epochs"]):
                b_inds = torch.randperm(
                    self.agent_cfg["batch_size"], device=self.device
                )
                for start in range(
                    0, self.agent_cfg["batch_size"], self.agent_cfg["minibatch_size"]
                ):
                    end = start + self.agent_cfg["minibatch_size"]
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.agent_cfg["clip_coef"])
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.agent_cfg["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - self.agent_cfg["clip_coef"],
                        1 + self.agent_cfg["clip_coef"],
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.agent_cfg["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.agent_cfg["clip_coef"],
                            self.agent_cfg["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.agent_cfg["ent_coef"] * entropy_loss
                        + v_loss * self.agent_cfg["vf_coef"]
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.agent_cfg["max_grad_norm"]
                    )
                    self.optimizer.step()

                if self.agent_cfg["target_kl"] is not None:
                    if approx_kl > self.agent_cfg["target_kl"]:
                        break

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar(
                "charts/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            #print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

            metrics.update({
                "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "charts/SPS": int(global_step / (time.time() - start_time)),
            })

            wandb.log(metrics)

        # envs.close()
        self.writer.close()

    def test(self):
        raise NotImplemented
