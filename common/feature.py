import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


class pm_feature:
    def __init__(
        self,
        env_cfg,
        combination=[True, True, True, True],
    ) -> None:
        self.env_cfg = env_cfg
        self.envdim = env_cfg["dim"]

        self._pos_norm = combination[0]
        self._vel_err = combination[1]
        self._vel_norm = combination[2]
        self._prox = combination[3]

        self.dim = (
            combination[0]  # pos
            + self.envdim * combination[1]  # vel_err
            + combination[2]  # vel_norm
            + combination[3]  # prox
        )

    def extract(self, s):
        features = []

        Kp = self.env_cfg["goal_lim"]
        Kv = self.env_cfg["vel_lim"]
        prox_thresh = self.env_cfg["task"]["proximity_threshold"]

        pos = s[:, 0 : self.envdim]
        posSquaredNorm = torch.linalg.norm(pos, axis=1, keepdims=True) ** 2
        posNormFeature = (
            torch.exp(-7.2 * posSquaredNorm / Kp**2)
            + torch.exp(-360 * posSquaredNorm / Kp**2)
        ) / 2
        if self._pos_norm:
            features.append(posNormFeature)

        vel = s[:, self.envdim : 2 * self.envdim]
        if self._vel_err:
            vel_feature = torch.exp(-16.0 * vel**2 / Kv**2)
            features.append(vel_feature)

        if self._vel_norm:
            velSquaredNorm = s[:, 4:5] ** 2
            velNormFeature = torch.exp(-16.0 * velSquaredNorm / Kv**2)
            features.append(velNormFeature)

        if self._prox:
            A = 1 / np.exp(-12.8 * prox_thresh**2 / Kp**2)
            proximity_rew_gauss = A * torch.exp(-12.8 * posSquaredNorm / Kp**2)
            proximity_rew = torch.where(
                posSquaredNorm > prox_thresh**2, proximity_rew_gauss, 1
            )
            features.append(proximity_rew)

        return torch.cat(features, 1)


class pointer_feature:
    def __init__(
        self,
        env_cfg,
        combination=[True, True, True, True],
    ) -> None:
        self.env_cfg = env_cfg
        self.envdim = env_cfg["dim"]

        self._px = combination[0]
        self._py = combination[1]
        self._vel_norm = combination[2]
        self._ang_norm = combination[3]
        self._angvel_norm = combination[4]

        self.dim = (
            combination[0]  # pos_norm
            + combination[1]  # vel_norm
            + combination[2]  # ang_norm
            + combination[3]  # angvel_norm
            + combination[4]
        )

    def extract(self, s):
        features = []

        # pos = s[:, 3:5]
        # pos_norm = torch.linalg.norm(pos, axis=1, keepdims=True)

        # if self._prox:
        #     features.append(-pos_norm)

        # vel = s[:, 5:7]
        # if self._vel_norm:
        #     features.append(-torch.linalg.norm(vel, axis=1, keepdims=True))

        # ang = s[:, 0:1]
        # if self._ang_norm:
        #     features.append(-torch.linalg.norm(ang, axis=1, keepdims=True))

        # angvel = s[:, 7:8]
        # if self._angvel_norm:
        #     features.append(-torch.linalg.norm(angvel, axis=1, keepdims=True))

        Kp = self.env_cfg["goal_lim"]
        Kv = self.env_cfg["vel_lim"]
        Ka = np.pi
        prox_thresh = self.env_cfg["task"]["proximity_threshold"]

        pxSquaredNorm = s[:, 3:4] ** 2
        if self._px:
            A = 1 / np.exp(-25 * prox_thresh**2 / Kp**2)
            prox_x_rew_gauss = A * torch.exp(-25 * pxSquaredNorm / Kp**2)
            prox_x_rew = torch.where(
                pxSquaredNorm > prox_thresh**2, prox_x_rew_gauss, 1
            )
            features.append(prox_x_rew)

        pySquaredNorm = s[:, 4:5] ** 2
        if self._py:
            A = 1 / np.exp(-25 * prox_thresh**2 / Kp**2)
            prox_y_rew_gauss = A * torch.exp(-25 * pySquaredNorm / Kp**2)
            prox_y_rew = torch.where(
                pySquaredNorm > prox_thresh**2, prox_y_rew_gauss, 1
            )
            features.append(prox_y_rew)

        # if self._px:
        #     pxSquaredNorm = s[:, 3:4] ** 2
        #     pxNormFeature = torch.exp(-25 * pxSquaredNorm / Kp**2)
        #     features.append(pxNormFeature)

        # if self._py:
        #     pySquaredNorm = s[:, 4:5] ** 2
        #     pyNormFeature = torch.exp(-25 * pySquaredNorm / Kp**2)
        #     features.append(pyNormFeature)

        if self._vel_norm:
            vel = s[:, 5:7]
            velSquaredNorm = torch.linalg.norm(vel, axis=1, keepdims=True) ** 2
            # velSquaredNorm = s[:, 9:10] ** 2
            velNormFeature = torch.exp(-30 * velSquaredNorm / Kv**2)
            features.append(velNormFeature)

        if self._ang_norm:
            angSquaredNorm = s[:, 0:1] ** 2
            angNormFeature = torch.exp(-50 * angSquaredNorm / Ka**2)
            features.append(angNormFeature)

        if self._angvel_norm:
            avelSquaredNorm = s[:, 7:8] ** 2
            avelNormFeature = torch.exp(-50 * avelSquaredNorm / Kv**2)
            features.append(avelNormFeature)

        return torch.concatenate(features, 1)
