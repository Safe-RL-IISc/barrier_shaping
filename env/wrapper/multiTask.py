from env import env_map
from common.feature import pm_feature, pointer_feature
import torch


class MultiTaskEnv:
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.env = env_map[env_cfg["env_name"]](env_cfg)

        self.task_w = env_cfg["task"]["task_w"]
        self.task_w_eval = env_cfg["task"]["task_w_eval"]

    def define_tasks(self, env_cfg, combination):
        def get_w(c, d, w):
            # feature order: [pos, pos_norm, vel, vel_norm, ang, angvel, success]

            if "pointmass" in env_cfg["env_name"].lower():
                w_pos_norm = c[0] * [w[0]]
                w_vel = c[1] * d * [w[1]]
                w_vel_norm = c[2] * [w[2]]
                w_prox = c[3] * [w[3]]
                return w_pos_norm + w_vel + w_vel_norm + w_prox

            elif "pointer" in env_cfg["env_name"].lower():
                w_px = c[0] * [w[0]]
                w_py = c[1] * [w[1]]
                w_vel_norm = c[2] * [w[2]]
                w_ang_norm = c[3] * [w[3]]
                w_angvel_norm = c[4] * [w[4]]

                return w_px + w_py + w_vel_norm + w_ang_norm + w_angvel_norm
            else:
                raise NotImplementedError(f"env name {env_cfg['env_name']} invalid")

        dim = env_cfg["dim"]

        task_w = get_w(combination, dim, self.task_w)
        task_w_eval = get_w(combination, dim, self.task_w_eval)

        tasks_train = torch.tensor(task_w, device="cuda:0", dtype=torch.float32)
        tasks_eval = torch.tensor(task_w_eval, device="cuda:0", dtype=torch.float32)

        return (tasks_train, tasks_eval)

    def getEnv(self):
        if "pointer" or "pointmass" not in self.env_cfg["env_name"].lower():
            return self.env, None, None
        feature_type = self.env_cfg["feature"]["type"]
        combination = self.env_cfg["feature"][feature_type]
        task_w = self.define_tasks(self.env_cfg, combination)
        if "pointer" in self.env_cfg["env_name"].lower():
            feature = pointer_feature(self.env_cfg, combination)
        elif "pointmass" in self.env_cfg["env_name"].lower():
            feature = pm_feature(self.env_cfg, combination)
        else:
            raise NotImplementedError(f'no such env {self.env_cfg["env_name"]}')

        return self.env, task_w, feature
