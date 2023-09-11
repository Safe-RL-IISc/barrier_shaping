from re import A
from typing import Tuple
import torch
import numpy as np
import json
import torch.nn as nn
import collections.abc

from omegaconf import DictConfig, OmegaConf
from typing import Dict


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def fix_wandb(d):
    ret = {}
    d = dict(d)
    for k, v in d.items():
        if not "." in k:
            ret[k] = v
        else:
            ks = k.split(".")
            a = fix_wandb({".".join(ks[1:]): v})
            if ks[0] not in ret:
                ret[ks[0]] = {}
            update_dict(ret[ks[0]], a)
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_sa_pairs(s: torch.tensor, a: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """s is state, a is action particles
    pair every state with each action particle

    for example, 2 samples of state and 3 action particls
    s = [s0, s1]
    a = [a0, a1, a2]

    s_tile = [s0, s1, s0, s1, s0, s1]
    a_tile = [a0, a1, a2, a0, a1, a2]

    Args:
        s (torch.tensor): (number of samples, state dimension)
        a (torch.tensor): (number of particles, action dimension)

    Returns:
        Tuple[torch.tensor, torch.tensor]:
            s_tile (n_sample*n_particles, state_dim)
            a_tile (n_sample*n_particles, act_dim)
    """
    n_particles = a.shape[0]
    n_samples = s.shape[0]
    state_dim = s.shape[1]

    s_tile = torch.tile(s, (1, n_particles))
    s_tile = s_tile.reshape(-1, state_dim)

    a_tile = torch.tile(a, (n_samples, 1))
    return s_tile, a_tile


def pile_sa_pairs(
    s: torch.tensor, a: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:
    """s is state, a is action particles
    pair every state with each action particle

    Args:
        s (tensor): (number of samples, state dimension)
        a (tensor): (number of samples, number of particles, action dimension)

    Returns:
        Tuple[torch.tensor, torch.tensor]:
            s_tile (n_sample*n_particles, state_dim)
            a_tile (n_sample*n_particles, act_dim)
    """
    n_samples = s.shape[0]
    state_dim = s.shape[1]
    n_particles = a.shape[1]
    act_dim = a.shape[2]

    s_tile = torch.tile(s, (1, n_particles))
    s_tile = s_tile.reshape(-1, state_dim)

    a_tile = a.reshape(-1, act_dim)
    return s_tile, a_tile


def get_sah_pairs(
    s: torch.tensor, a: torch.tensor, h: torch.tensor
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    n_samples = a.shape[0]

    s_tile, a_tile = get_sa_pairs(s, a)
    h_tile = torch.tile(h, (n_samples, 1))
    return s_tile, a_tile, h_tile


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


def check_samples(obj):
    if obj.ndim > 1:
        n_samples = obj.shape[0]
    else:
        n_samples = 1
    return n_samples


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape
    assert len(tensor_shape) == len(
        expected_shape
    ), f"expect len a {len(tensor_shape)}, b {len(expected_shape)}"
    assert all(
        [a == b for a, b in zip(tensor_shape, expected_shape)][1:]
    ), f"expect shape a {tensor_shape}, b {expected_shape}"


def np2ts(obj: np.ndarray) -> torch.Tensor:
    if isinstance(obj, np.ndarray) or isinstance(obj, float):
        obj = torch.tensor(obj, dtype=torch.float32).to(device)
    return obj


def ts2np(obj: torch.Tensor) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().detach().numpy()
    return obj


def check_obs(obs, obs_dim):
    obs = np2ts(obs)
    n_samples = check_samples(obs)
    obs = obs.reshape(n_samples, obs_dim)
    return obs


def check_act(action, action_dim, type=np.float32):
    # action = ts2np(action)
    # n_samples = check_samples(action)
    # action = action.reshape(n_samples, action_dim).astype(type)
    return torch.clamp(action, min=-1, max=1)


def to_batch(
    state,
    feature,
    action,
    reward,
    next_state,
    done,
    device,
):
    state = torch.FloatTensor(state).to(device)
    feature = torch.FloatTensor(feature).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    done = torch.FloatTensor(done).to(device)
    return state, feature, action, reward, next_state, done


def to_batch_rnn(
    state,
    feature,
    action,
    reward,
    next_state,
    done,
    h_in0,
    h_out0,
    h_in1,
    h_out1,
    device,
):
    state, feature, action, reward, next_state, done = to_batch(
        state, feature, action, reward, next_state, done, device
    )
    return (
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        h_in0,
        h_out0,
        h_in1,
        h_out1,
    )


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def update_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def linear_schedule(cur_step, tot_step, schedule):
    progress = cur_step / tot_step
    return np.clip(
        progress * (schedule[1] - schedule[0]) + schedule[0],
        np.min(np.array(schedule)),
        np.max(np.array(schedule)),
    )


def dump_cfg(path, obj):
    with open(path, "w") as fp:
        json.dump(dict(obj), fp, default=lambda o: o.__dict__, indent=4, sort_keys=True)
