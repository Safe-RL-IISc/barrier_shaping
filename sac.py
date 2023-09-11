import torch
from torch.optim import Adam

import wandb
from common.agent import IsaacAgent
from common.policy import GaussianPolicy
from common.util import grad_false, hard_update, soft_update, update_params
from common.value_function import TwinnedQNetwork


class SACAgent(IsaacAgent):
    """SAC
    Tuomas Haarnoja, Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
    see https://github.com/haarnoja/sac/blob/master/sac/algos/sac.py
    and https://github.com/ku2482/soft-actor-critic.pytorch
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lr = self.agent_cfg["lr"]
        self.policy_lr = self.agent_cfg["policy_lr"]
        self.value_net_kwargs = self.agent_cfg["value_net_kwargs"]
        self.policy_net_kwargs = self.agent_cfg["policy_net_kwargs"]
        self.gamma = self.agent_cfg["gamma"]
        self.tau = self.agent_cfg["tau"]
        self.td_target_update_interval = int(
            self.agent_cfg["td_target_update_interval"]
        )
        self.updates_per_step = self.agent_cfg["updates_per_step"]
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.entropy_tuning = self.agent_cfg["entropy_tuning"]
        self.droprate = self.value_net_kwargs["droprate"]

        self.critic = TwinnedQNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            **self.value_net_kwargs,
        ).to(self.device)

        self.critic_target = TwinnedQNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            **self.value_net_kwargs,
        ).to(self.device)
        if self.droprate <= 0.0:
            self.critic_target = self.critic_target.eval()

        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

        self.policy = GaussianPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            **self.policy_net_kwargs,
        ).to(self.device)

        # self.q1_optimizer = Adam(self.critic.Q1.parameters(), lr=self.lr)
        # self.q2_optimizer = Adam(self.critic.Q2.parameters(), lr=self.lr)
        self.q_optimizer = Adam(
            self.critic.parameters(), lr=self.lr, betas=[0.9, 0.999]
        )
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=self.policy_lr, betas=[0.9, 0.999]
        )

        if self.entropy_tuning:
            self.alpha_lr = self.agent_cfg["alpha_lr"]
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_shape).to(self.device)
            ).item()  # target entropy = -|A|
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device
            )  # optimize log(alpha), instead of alpha
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(self.agent_cfg["alpha"]).to(self.device)

        self.learn_steps = 0

    def explore(self, s, w):  # act with randomness
        with torch.no_grad():
            a, _, _ = self.policy.sample(s)
        return a

    def exploit(self, s, w):  # act without randomness
        with torch.no_grad():
            _, _, a = self.policy.sample(s)
        return a

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # if self.per:
        #     batch, indices, weights = self.replay_buffer.sample(self.mini_batch_size)
        # else:
        batch = self.replay_buffer.sample(self.mini_batch_size)
        weights = 1

        q_loss, errors, mean_q1 = self.update_critic(batch, weights)
        policy_loss, entropies = self.update_policy(batch, weights)

        # update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)
        # update_params(self.q1_optimizer, self.critic.Q1, q1_loss, self.grad_clip)
        # update_params(self.q2_optimizer, self.critic.Q2, q2_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        # if self.per:
        #     self.replay_buffer.update_priority(indices, errors.cpu().numpy())

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/Q": q_loss,
                "loss/policy": policy_loss,
                "state/mean_Q1": mean_q1,
                "state/entropy": entropies.detach().mean().item(),
            }
            if self.entropy_tuning:
                metrics.update(
                    {
                        "loss/alpha": entropy_loss.detach().item(),
                        "state/alpha": self.alpha.mean().detach().item(),
                    }
                )

            wandb.log(metrics)

    def update_critic(self, batch, weights):
        (s, f, a, r, s_next, dones) = batch

        curr_q1, curr_q2 = self.calc_current_q(s, a)
        target_q = self.calc_target_q(r, s_next, dones)

        # Critic loss is mean squared TD errors.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # log values to monitor training.
        q_loss = q_loss.detach().item()
        mean_q1 = curr_q1.detach().mean().item()

        return q_loss, errors, mean_q1

    def update_policy(self, batch, weights):
        (s, f, a, r, s_next, dones) = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_a, entropy, _ = self.policy.sample(s)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(s, sampled_a)

        if self.droprate > 0.0:
            q = 0.5 * (q1 + q2)
        else:
            q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = torch.mean((-q - self.alpha * entropy) * weights)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        return policy_loss.detach().item(), entropy

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach() * weights
        )
        return entropy_loss

    def calc_current_q(self, s, a):
        curr_q1, curr_q2 = self.critic(s, a)
        return curr_q1, curr_q2

    def calc_target_q(self, r, s_next, dones):
        with torch.no_grad():
            a_next, _, _ = self.policy.sample(s_next)
            next_q1, next_q2 = self.critic_target(s_next, a_next)
            next_q = torch.min(next_q1, next_q2)

        target_q = r + (~dones) * self.gamma * next_q
        return target_q

    def calc_priority_error(self, batch):
        (s, _, a, r, s_next, dones) = batch
        with torch.no_grad():
            curr_q1, curr_q2 = self.calc_current_q(s, a)
        target_q = self.calc_target_q(r, s_next, dones)
        error = torch.abs(curr_q1 - target_q).cpu().numpy()
        return error

    def save_torch_model(self):
        from pathlib import Path

        path = self.log_path + f"model{self.episodes}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        self.policy.save(path + "policy")
        self.critic.Q1.save(path + "critic1")
        self.critic.Q2.save(path + "critic2")

    def load_torch_model(self, path):
        self.policy.load(path + "policy")
        self.critic.Q1.load(path + "critic1")
        self.critic.Q2.load(path + "critic2")
        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)
