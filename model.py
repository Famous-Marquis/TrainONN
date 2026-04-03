import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from lightning import LightningModule
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
    matrix_layer = torch.tensor((), dtype=torch.complex64).cuda()  # init null layer matrix
    for i in range(phs_layer.shape[0] // 2):
        # matrix = (torch.tensor([[torch.exp(1j * phs_layer[2 * i]), 0],
        #                         [0, torch.exp(1j * phs_layer[2 * i + 1])]], dtype=torch.complex64)
        #           @ torch.tensor([[np.sqrt(2) / 2, np.sqrt(2) / 2 * 1j],
        #                           [np.sqrt(2) / 2 * 1j, np.sqrt(2) / 2], ], dtype=torch.complex64))
        matrix = (math.sqrt(2) / 2 * torch.ones((2, 2), dtype=torch.complex64, device='cuda')
                  @ torch.diag(-1j * torch.ones((2,), dtype=torch.complex64, device='cuda'))
                  @ torch.diag(torch.exp(1j * phs_layer[2 * i:2 * i + 2])))
        matrix_layer = torch.block_diag(matrix_layer, matrix)
    matrix_layer = matrix_layer[1:]  # delete init null matrix
    # print(matrix_layer)
    return matrix_layer


def mzi_mesh(global_phi, parallel=8):
    matrix_mesh = torch.diag(torch.ones(parallel, dtype=torch.complex64, device='cuda'))
    for lay_idx in range(global_phi.shape[0]):
        layer_phi = global_phi[lay_idx]
        if lay_idx in [2, 4, 6, 8]:
            matrix_layer = symmetric_mzi_matrix(layer_phi[1:-1])
            matrix_layer = torch.block_diag(torch.ones((1,), dtype=torch.complex64, device='cuda'),
                                            matrix_layer,
                                            torch.ones((1,), dtype=torch.complex64, device='cuda'))
            matrix_mesh = matrix_mesh @ matrix_layer
        else:
            matrix_layer = symmetric_mzi_matrix(layer_phi)
            matrix_mesh = matrix_mesh @ matrix_layer
    return matrix_mesh


class SimMZIMitrix(MZIMatrix):
    def __init__(self, layer_num, parallel):
        super().__init__()
        self.layer_num = layer_num
        self.parallel = parallel
        self.global_phi = torch.randn((layer_num, parallel),
                                      dtype=torch.float32).cuda()  # [layer,parallel], random init
        self.mesh_matrix = mzi_mesh(self.global_phi)
        self.shifter_curve = {"a": torch.randn((layer_num, parallel), dtype=torch.float32).cuda() * 0,
                              "b": torch.ones((layer_num, parallel), dtype=torch.float32).cuda(),
                              "c": torch.randn((layer_num, parallel), dtype=torch.float32).cuda()}
        self.global_phi = torch.tensor(self.global_phi, dtype=torch.float32).cuda() \
            if not isinstance(self.global_phi,torch.Tensor) \
            else self.global_phi.detach().clone().to(dtype=torch.float32).cuda()
        self.nonlinear = torch.square
        # todo obs输入非负，mzi后跟的nonlinear应该在非负区域非线性
        # model free 的non linear
        # assert shifter curve: phi=ax^2+bx+c (rad)

    def step(self, x, y_target):
        y_theta = self.forward(x)
        obs = torch.cat((x, y_target - y_theta), dim=0)
        return obs, y_theta

    def forward(self, x):
        """get amp and phase result"""
        x = x.to(dtype=torch.complex64)
        detected = (x @ self.mesh_matrix)
        return self.nonlinear(detected).float()

    def update_voltage(self, v_diff):
        """v: voltage, unit: Volt
            v has no limit
        """


        v_diff = v_diff.view(self.layer_num, self.parallel)
        assert len(v_diff.shape)==2,"v_diff shape wrong! Damn"
        updated_phase = (self.shifter_curve["a"] * v_diff ** 2
                         + self.shifter_curve["b"] * v_diff
                         + self.shifter_curve["c"])
        self.global_phi = updated_phase + self.global_phi
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

class DNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.model(x)


class PPO(LightningModule):
    def __init__(self, NN, batch, act_space, obs_space, actor_lr, critic_lr, mzi, epsilon,gamma=0.95,n_updates_per_batch=5):
        """

        :param act_space: tap to control [layer_num * parallel]
        :param obs_space: MZI inp and oup [parallel*2,] X cat [Y_target - Y_theta]
        """
        super().__init__()
        self.automatic_optimization = False
        self.env = mzi
        self.batch = batch
        self.epsilon = epsilon
        self.act_space = act_space
        self.obs_space = obs_space
        self.random_vector = ...  # a random vector
        self.actor = NN(obs_space, act_space)
        self.critic = NN(obs_space, 1)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.n_updates_per_batch=n_updates_per_batch
        # todo critic 输入包含y target吗？ 是的，x,y_target-y_theta
        self.loss_fn = torch.nn.functional.mse_loss
        self.max_step_per_episode = 100

        self.cov_var = torch.full((act_space,), 0.10, dtype=torch.float32).cuda()
        # todo cov_var 可变
        self.cov_mat = torch.diag(self.cov_var)
        self.save_hyperparameters(ignore=['NN', 'mzi'])

    def configure_optimizers(self):
        # 2. 在这里配置优化器，Lightning 会自动接管它们（包括混合精度、设备转移等）
        actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        # 返回列表
        return actor_optim, critic_optim

    def evaluate(self, episode_obs, episode_acts):
        V = self.critic(episode_obs)
        mean = self.actor(episode_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(episode_acts)
        return V, log_probs

    def get_action(self, batch_obs):
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        batch_action = dist.sample()
        log_probs = dist.log_prob(batch_action)
        return batch_action.detach(), log_probs.detach()

    def calculate_rwtg(self, episode_rewards, normalized=False):
        """:return discounted_rwtg [episode,]"""
        T = len(episode_rewards)
        rwtg = torch.zeros_like(episode_rewards)

        running_add = torch.zeros_like(episode_rewards[0])
        for t in reversed(range(T)):
            running_add = episode_rewards[t] + self.gamma * running_add
            rwtg[t] = running_add
        return rwtg

    def training_step(self, single_data_pair):
        """FFM"""
        actor_optim, critic_optim = self.optimizers()
        x, y_target = single_data_pair  # x:[parallel],y:[parallel], random vector indicating true or false
        with torch.no_grad():
            episode_obs, episode_acts, episode_log_probs, episode_rwtg = self.rollout(x, y_target)
            """shape: [steps_per_episode,batch,(parallel)]"""
            V, _ = self.evaluate(episode_obs, episode_acts, )
            # A_k=(batch_rtgs-batch_rtgs.mean())/(batch_rtgs.std()+1e-10)
            episode_A_k = episode_rwtg - V.detach().squeeze(-1)
            episode_A_k = (episode_A_k - episode_A_k.mean()) / (episode_A_k.std() + 1e-10)
            # Optional: A_k normalize `A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)`
        for i in range(self.n_updates_per_batch):
            V, curr_log_probs = self.evaluate(episode_obs, episode_acts)
            ratios = torch.exp(curr_log_probs - episode_log_probs)

            surr1 = ratios * episode_A_k
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * episode_A_k

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.mse_loss(V.squeeze(), episode_rwtg)
            assert len(actor_loss.shape) == 0, "loss is not scalar"

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            # todo: log
            self.log("train_actor_loss", actor_loss.item(), prog_bar=True)
            self.log("train_critic_loss", critic_loss.item(), prog_bar=True)

    def rollout(self, x, y_target):
        """以x，跑一个回合"""
        episode_obs = torch.zeros((self.max_step_per_episode, self.obs_space), dtype=torch.float32).cuda()
        episode_acts = torch.zeros((self.max_step_per_episode, self.act_space), dtype=torch.float32).cuda()
        episode_log_probs = torch.zeros((self.max_step_per_episode,), dtype=torch.float32).cuda()
        episode_rewards = torch.zeros((self.max_step_per_episode,), dtype=torch.float32).cuda()
        obs, _ = self.env.step(x, y_target)
        for t in range(self.max_step_per_episode):
            act, log_prob = self.get_action(obs)
            self.env.update_voltage(act)
            obs, y_theta = self.env.step(x,y_target)
            reward = -self.loss_fn(y_theta, y_target)

            # Note! no terminated
            episode_obs[t] = obs
            episode_acts[t] = act
            episode_rewards[t] = reward
            episode_log_probs[t] = log_prob
        rwtg = self.calculate_rwtg(episode_rewards).cuda()

        return episode_obs, episode_acts, episode_log_probs, rwtg
        # todo 用register_buffer 注册不需要求梯度的常量
        # todo 各个代码中，不要将cuda写死，用self device

if __name__ == '__main__':
    symmetric_mzi_matrix(np.ones((6,)))
    matrix_mesh = mzi_mesh(np.ones((10, 8)))
    print(matrix_mesh.shape)
