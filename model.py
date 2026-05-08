import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
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
        self.nonlinear = torch.nn.functional.sigmoid
        # todo obs输入非负，mzi后跟的nonlinear应该在非负区域非线性
        # model free 的non linear
        # assert shifter curve: phi=ax^2+bx+c (rad)

    def step(self, x):
        y_theta = self.forward(x)
        # obs = torch.cat((x, y_target - y_theta), dim=0)
        return y_theta

    def forward(self, x):
        """get amp and phase result"""
        x = x.to(dtype=torch.complex64)
        detected = (x @ self.mesh_matrix).abs().float()
        return self.nonlinear(detected)

    def update_voltage(self, v_diff):
        """v: voltage, unit: Volt
            v has no limit
        """


        v_diff = v_diff.view(self.layer_num, self.parallel)
        assert len(v_diff.shape)==2,"v_diff shape wrong! Damn"
        updated_phase = (self.shifter_curve["a"] * v_diff ** 2
                         + self.shifter_curve["b"] * v_diff)
        self.global_phi = updated_phase
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


import torch
import torch.nn as nn
from lightning import LightningModule
from torch.distributions import MultivariateNormal


class PPOFixedMZI(LightningModule):
    def __init__(self, M_samples, act_space, actor_lr, mzi, epsilon=0.2, n_updates_per_batch=5):
        super().__init__()
        self.automatic_optimization = False
        self.env = mzi

        self.M_samples = M_samples  # 对应论文中的 M：每次采样多少组不同的 MZI 参数去试错
        self.act_space = act_space
        self.epsilon = epsilon
        self.actor_lr = actor_lr
        self.n_updates_per_batch = n_updates_per_batch

        # 【关键改变1】：Actor 不再是神经网络，而是直接的物理参数分布的均值 (mu)
        # 初始化为你猜测的一组电压，或者全 0
        self.mu = nn.Parameter(torch.zeros(act_space, dtype=torch.float32))

        # 设定一个固定的方差 (论文中设定为 0.04) 或者也可以设为可学习的 Parameter
        cov_var = torch.full((act_space,), 0.04, dtype=torch.float32)
        self.register_buffer('cov_mat', torch.diag(cov_var))

        self.loss_fn = torch.nn.functional.cross_entropy

    def configure_optimizers(self):
        # 优化器现在只优化 mu 这一个张量
        actor_optim = torch.optim.Adam([self.mu], lr=self.actor_lr)
        return actor_optim

    def get_action_and_logprob(self, action=None):
        # 构建正态分布 N(mu, sigma^2)
        dist = MultivariateNormal(self.mu, self.cov_mat)
        if action is None:
            # 采样
            action = dist.sample()
        # 计算 log_prob
        log_prob = dist.log_prob(action)
        return action, log_prob
    def validation_step(self,batch_data, batch_idx):
        x_batch, y_target_batch = batch_data
        y_theta_batch = self.env.step(x_batch)
        y_theta_batch=torch.argmax(y_theta_batch, dim=1)
        batch_acc=torch.sum(y_theta_batch==y_target_batch)/y_target_batch.size(0)
        self.log("val_acc", batch_acc.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    def training_step(self, batch_data):
        # batch_data: 包含了一批输入信号 X 和对应的 Y_target [batch_size, parallel]
        x_batch, y_target_batch = batch_data
        actor_optim = self.optimizers()

        # 1. 采样 M 组不同的 MZI 动作 (对应论文 Step 1)
        # shape: [M_samples, act_space]
        sampled_actions = torch.zeros((self.M_samples, self.act_space), device=self.device)
        old_log_probs = torch.zeros(self.M_samples, device=self.device)
        rewards = torch.zeros(self.M_samples, device=self.device)

        with torch.no_grad():
            for i in range(self.M_samples):
                action, log_prob = self.get_action_and_logprob()
                sampled_actions[i] = action
                old_log_probs[i] = log_prob

                # 2. 物理评估 (对应论文 Step 2)
                # 将采样到的这组 action（电压）写入 MZI
                self.env.update_voltage(action)

                # 用这*一组*物理参数，跑完*一整批*的数据 X，得到输出
                y_theta_batch = self.env.step(x_batch)
                # y_theta_batch = torch.argmax(y_theta_batch, dim=1)

                # 3. 计算这一组参数在这批数据上的总表现 (对应论文 Step 3)
                # reward 是一个标量
                reward = -self.loss_fn(y_theta_batch, y_target_batch)
                rewards[i] = reward

            # 计算优势函数 Advantage
            # 因为没有序列决策（没有时间步），所以不需要计算 rwtg 和 Critic 网络，直接用 Reward 标准化即可
            # A_k = (R - R_mean) / (R_std + 1e-10)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # 4. PPO 数字策略更新 (对应论文 Step 4)
        for _ in range(self.n_updates_per_batch):
            # 重新计算当前 mu 下，那些 sampled_actions 的概率
            _, curr_log_probs = self.get_action_and_logprob(sampled_actions)

            ratios = torch.exp(curr_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # 我们要最大化 reward，所以 loss 加负号
            actor_loss = -torch.min(surr1, surr2).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

        self.log("train_actor_loss", actor_loss.item(), prog_bar=True)
        self.log("batch_mean_reward", rewards.mean().item(), prog_bar=True)


# 模型1：你的方案（8x8 + ReLU）
class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)  # 8x8矩阵
        self.activation = nn.ReLU()
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    def validation_step(self, batch_data):
        x_batch, y_target_batch = batch_data
        y_theta_batch = self.forward(x_batch)
        y_theta_batch = torch.argmax(y_theta_batch, dim=1)
        batch_acc = torch.sum(y_theta_batch==y_target_batch)/y_target_batch.size(0)
        self.log("val_acc", batch_acc.item(), on_step=False, on_epoch=True, prog_bar=True)


    def training_step(self, batch_data):
        x_batch, y_target_batch = batch_data
        y_theta_batch = self.forward(x_batch)
        loss = torch.nn.functional.cross_entropy(y_theta_batch, y_target_batch)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def forward(self, x):
        return self.activation(self.linear(x))


if __name__ == '__main__':
    symmetric_mzi_matrix(np.ones((6,)))
    matrix_mesh = mzi_mesh(np.ones((10, 8)))
    print(matrix_mesh.shape)
