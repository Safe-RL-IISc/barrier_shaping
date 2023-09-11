import torch


class VectorizedReplayBuffer:
    def __init__(self, obs_shape, action_shape, feature_shape, capacity, device):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device

        self.obses = torch.empty(
            (capacity, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.features = torch.empty(
            (capacity, *feature_shape), dtype=torch.float32, device=self.device
        )
        self.next_obses = torch.empty(
            (capacity, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty(
            (capacity, *action_shape), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.empty(
            (capacity, 1), dtype=torch.float32, device=self.device
        )
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)

        self.capacity = capacity
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.idx

    def add(self, obs, feature, action, reward, next_obs, done):
        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0:overflow] = obs[-overflow:]
            self.features[0:overflow] = feature[-overflow:]
            self.actions[0:overflow] = action[-overflow:]
            self.rewards[0:overflow] = reward[-overflow:]
            self.next_obses[0:overflow] = next_obs[-overflow:]
            self.dones[0:overflow] = done[-overflow:]
            self.full = True
        self.obses[self.idx : self.idx + remaining_capacity] = obs[:remaining_capacity]
        self.features[self.idx : self.idx + remaining_capacity] = feature[
            :remaining_capacity
        ]
        self.actions[self.idx : self.idx + remaining_capacity] = action[
            :remaining_capacity
        ]
        self.rewards[self.idx : self.idx + remaining_capacity] = reward[
            :remaining_capacity
        ]
        self.next_obses[self.idx : self.idx + remaining_capacity] = next_obs[
            :remaining_capacity
        ]
        self.dones[self.idx : self.idx + remaining_capacity] = done[:remaining_capacity]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        """

        idxs = torch.randint(
            0,
            self.capacity if self.full else self.idx,
            (batch_size,),
            device=self.device,
        )
        obses = self.obses[idxs]
        features = self.features[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        dones = self.dones[idxs]

        # (s, f, a, r, s_next, dones) = batch

        return (obses, features, actions, rewards, next_obses, dones)
