from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn.functional import batch_norm


class MZIMatrix(ABC):
    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def update_voltage(self, v_diff):
        ...

    @abstractmethod
    def hardware_miss_alignment(self, eps):
        ...

    @abstractmethod
    def temperature_shift(self, T):
        ...

    @abstractmethod
    def power_loss(self, power_loss):
        ...


class OptimizerAgent(ABC):
    @abstractmethod
    def action(self, state):
        """state: x,y_theta"""
        ...


def symmetric_mzi_matrix(phs_layer):
    assert phs_layer.shape[0] % 2 == 0, "phs_layer must be even"
    matrix_layer = torch.tensor((), dtype=torch.complex64)  # init null layer matrix
    for i in range(phs_layer.shape[0] // 2):
        matrix = (torch.tensor([[np.exp(1j * phs_layer[2 * i]), 0],
                                [0, np.exp(1j * phs_layer[2 * i + 1])]], dtype=torch.complex64)
                  @ torch.tensor([[np.sqrt(2) / 2, np.sqrt(2) / 2 * 1j],
                                  [np.sqrt(2) / 2 * 1j, np.sqrt(2) / 2], ], dtype=torch.complex64))
        matrix_layer = torch.block_diag(matrix_layer, matrix)
    matrix_layer = matrix_layer[1:]  # delete init null matrix
    # print(matrix_layer)
    return matrix_layer


def mzi_mesh(global_phi, parallel=8):
    matrix_mesh = torch.diag(torch.ones(parallel, dtype=torch.complex64))
    for lay_idx, layer_phi in enumerate(global_phi):
        if lay_idx in [2, 4, 6, 8]:
            matrix_layer = symmetric_mzi_matrix(layer_phi[1:-1])
            matrix_layer = torch.block_diag(torch.ones((1,), dtype=torch.complex64),
                                            matrix_layer,
                                            torch.ones((1,), dtype=torch.complex64))
            matrix_mesh = matrix_layer @ matrix_mesh
        else:
            matrix_layer = symmetric_mzi_matrix(layer_phi)
            matrix_mesh = matrix_layer @ matrix_mesh
    return matrix_mesh


class SimMZIMitrix(MZIMatrix):
    def __init__(self, layer_num, parallel):
        super().__init__()
        self.global_phi = np.random.randn(layer_num, parallel).astype(np.complex64)  # [layer,parallel], random init
        self.mesh_matrix = mzi_mesh(self.global_phi)
        self.shifter_curve = {"a": np.random.randn(layer_num, parallel),
                              "b": np.random.randn(layer_num, parallel),
                              "c": np.random.randn(layer_num, parallel)}
        # assert shifter curve: phi=ax^2+bx+c

    def forward(self, x):
        """get amp and phase result"""
        return self.mesh_matrix @ x

    def update_voltage(self, v_diff):
        """v: voltage, unit: Volt"""
        updated_phase = (self.shifter_curve["a"] * v_diff ** 2
                         + self.shifter_curve["b"] * v_diff
                         + self.shifter_curve["c"])
        self.global_phi += updated_phase
        self.mesh_matrix = mzi_mesh(self.global_phi)

    def hardware_miss_alignment(self, eps):
        # todo
        return None

    def temperature_shift(self, T):
        # todo
        return None

    def power_loss(self, power_loss):
        # todo
        ...


# class RLEnv:
#     def __init__(self):
#         ...
#     def step(self,act,x,y_target):
#         self.mzi.update_voltage(act)
#         y_theta=self.mzi.forward(x)
#         obs=y_theta
#         return obs

class PPO:
    def __init__(self, NN, act_space, obs_space, actor_lr, critic_lr, mzi):
        """

        :param act_space: tap to control [layer_num * parallel]
        :param obs_space: MZI inp and oup [parallel*2,] X cat Y_theta
        """
        self.env = mzi  # mzi contains a random vector
        self.env.initialize()
        self.actor = NN(obs_space, act_space)
        self.critic = NN(obs_space, 1)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.cov_var = torch.full((act_space,), 0.5, dtype=torch.float32)
        self.cov_mat = torch.diag(self.cov_var)
        ...

    def evaluate(self, batch_obs, batch_acts):
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return log_probs
        ...  # todo

    def get_action(self, batch_obs):
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return log_probs

    def calculate_Ak(self, episode_rewards):
        batch_Ak = []
        discounted_Ak = 0
        for reward in episode_rewards:
            discounted_Ak = reward + discounted_Ak * self.gamma
            batch_Ak.insert(0, discounted_Ak)
        batch_Ak = torch.tensor(batch_Ak, dtype=torch.float32)
        return batch_Ak

    def training_step(self, batch):
        """FFM"""
        x, y_target = batch  # x:[batch,parallel],y:[batch,parallel], random vector indicating true or false
        batch_obs, batch_acts, batch_log_probs, A_k = self.rollout(x, y_target)

        # log_probs=self.evaluate(batch_obs,batch_acts)
        # A_k=(batch_rtgs-batch_rtgs.mean())/(batch_rtgs.std()+1e-10)

        # Optional: A_k normalize `A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)`
        for i in range(self.n_updates_per_batch):
            curr_log_probs = self.evaluate(batch_obs, batch_acts)
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * A_k

            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()
            # todo: log

    def rollout(self, x, y_target):
        """以x，跑一个回合"""
        episode_obs, episode_acts, episode_log_probs, episode_rtgs = [], [], [], []
        episode_rewards = []
        obs = self.env.forward(x)
        for t in range(self.max_step_per_episode):
            act, log_prob = self.get_action(obs)
            self.env.update(act)
            obs = self.env.farward(x)
            reward = self.loss_fn(obs, y_target)

            # Note! no terminated
            episode_obs.append(obs)
            episode_acts.append(act)
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
        episode_obs = torch.tensor(episode_obs, dtype=torch.float32)
        episode_acts = torch.tensor(episode_acts, dtype=torch.float32)
        episode_log_probs = torch.tensor(episode_log_probs, dtype=torch.float32)
        A_k = self.compute_Ak(episode_rewards)

        return episode_obs, episode_acts, episode_log_probs, A_k


if __name__ == '__main__':
    symmetric_mzi_matrix(np.ones((6,)))
    matrix_mesh = mzi_mesh(np.ones((10, 8)))
    print(matrix_mesh.shape)
