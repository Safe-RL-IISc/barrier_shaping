import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from common.util import omegaconf_to_dict, print_dict, fix_wandb, update_dict

from compose import CompositionAgent
from sac import SACAgent
from ppo import PPO_agent
from ppoh import PPOHagent

import torch
import numpy as np

import wandb


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    # wandb.init(mode="disabled")
    wandb.init()
    print(wandb.config, "\n\n")
    wandb_dict = fix_wandb(wandb.config)

    print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    # cfg_dict["env"]["episode_max_step"] = int(50 * (512 / cfg_dict["env"]["num_envs"]))
    print_dict(cfg_dict)

    torch.manual_seed(cfg_dict["seed"])
    np.random.seed(cfg_dict["seed"])

    if "sac" in cfg_dict["agent"]["name"].lower():
        agent = SACAgent(cfg=cfg_dict)
    elif "ppo" in cfg_dict["agent"]["name"].lower():
        agent = PPO_agent(cfg=cfg_dict)
    else:
        raise ValueError(f'{cfg_dict["agent"]["name"]} no implemented')

    agent.run()
    wandb.finish()

if __name__ == "__main__":
    launch_rlg_hydra()
