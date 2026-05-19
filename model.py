import math
from abc import ABC, abstractmethod
import torch.functional as F
from typing import Any

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
from torch.distributions import MultivariateNormal, Normal
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
    device = phs_layer.device  # 【关键1】：从输入的 phase layer 推断 device，不写死 cuda

    matrices = []
    # 【优化】：预先计算好常数矩阵，避免在循环中重复创建
    mat1 = (math.sqrt(2) / 2) * torch.ones((2, 2), dtype=torch.complex64, device=device)
    mat2 = torch.diag(-1j * torch.ones((2,), dtype=torch.complex64, device=device))
    base_mat = mat1 @ mat2

    for i in range(phs_layer.shape[0] // 2):
        mat3 = torch.diag(torch.exp(1j * phs_layer[2 * i: 2 * i + 2]))
        matrix = base_mat @ mat3
        matrices.append(matrix)

    # 【优化】：将列表解包一次性构建块对角矩阵，比循环里拼接更快
    matrix_layer = torch.block_diag(*matrices)
    return matrix_layer


def mzi_mesh(global_phi, parallel=8):
    device = global_phi.device  # 【关键1】
    matrix_mesh = torch.eye(parallel, dtype=torch.complex64, device=device)

    for lay_idx in range(global_phi.shape[0]):
        layer_phi = global_phi[lay_idx]
        if lay_idx in [2, 4, 6, 8]:
            matrix_layer = symmetric_mzi_matrix(layer_phi[1:-1])
            # padding 两端用 [1]
            padding = torch.ones((1,), dtype=torch.complex64, device=device)
            matrix_layer = torch.block_diag(padding, matrix_layer, padding)
            matrix_mesh = matrix_mesh @ matrix_layer
        else:
            matrix_layer = symmetric_mzi_matrix(layer_phi)
            matrix_mesh = matrix_mesh @ matrix_layer
    return matrix_mesh


class SimMZIMitrix(torch.nn.Module):  # 假设你继承自 nn.Module
    def __init__(self, layer_num, parallel):
        super().__init__()
        self.layer_num = layer_num
        self.parallel = parallel

        # 【关键2】：使用 register_buffer。这些张量会保存在模块的状态字典中，
        # 且调用 self.cuda() 或 self.to(device) 时，它们会自动转移到对应设备。
        self.register_buffer("global_phi", torch.randn((layer_num, parallel), dtype=torch.float32))

        # 将字典拆解为 buffer，原代码的 update 里面漏加了 c，我在这里修复了
        self.register_buffer("shifter_a", torch.zeros((layer_num, parallel), dtype=torch.float32))
        self.register_buffer("shifter_b", torch.ones((layer_num, parallel), dtype=torch.float32))
        self.register_buffer("shifter_c", torch.randn((layer_num, parallel), dtype=torch.float32))

        # 初始矩阵，不需要 registered，可以在 forward 中动态确保它在正确设备上
        self.mesh_matrix = mzi_mesh(self.global_phi)
        self.nonlinear = nn.Identity()# F.sigmoid

    def step(self, x):
        return self.forward(x)

    def forward(self, x):
        """get amp and phase result"""
        x = x.to(dtype=torch.complex64)

        # 确保计算图中的 mesh_matrix 和输入 x 在同一个 device 上
        if self.mesh_matrix.device != x.device:
            self.mesh_matrix = self.mesh_matrix.to(x.device)

        detected = (x @ self.mesh_matrix).abs().square().float()
        return self.nonlinear(detected)

    def update_voltage(self, v_diff):
        v_diff = v_diff.view(self.layer_num, self.parallel)
        assert len(v_diff.shape) == 2, "v_diff shape wrong! Damn"

        # 因为全都是 buffer，它们天然就在同一个 device 上
        updated_phase = (self.shifter_a * v_diff ** 2 +
                         self.shifter_b * v_diff +
                         self.shifter_c)  # 补充了 + self.shifter_c

        # 覆盖 global_phi，使用无梯度 in-place 更新
        self.global_phi.copy_(updated_phase)
        self.mesh_matrix = mzi_mesh(self.global_phi)
# class RLEnv:
#     def __init__(self):
#         ...
#     def step(self,act,x,y_target):
#         self.mzi.update_voltage(act)
#         y_theta=self.mzi.forward(x)
#         obs=y_theta
#         return obs
class TiledMZIMatrix(nn.Module):
    def __init__(self, in_features, out_features, layer_num, parallel=8, device="cuda:0"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_num = layer_num
        self.parallel = parallel

        self.in_blocks = math.ceil(in_features / parallel)
        self.out_blocks = math.ceil(out_features / parallel)

        self.pad_in = self.in_blocks * parallel - in_features
        self.pad_out = self.out_blocks * parallel - out_features

        param_shape = (self.in_blocks, self.out_blocks, layer_num, parallel)
        mzi_property_shape = (layer_num, parallel)

        # 1. 注册基础物理参数 buffer
        self.register_buffer("global_phi", torch.randn(param_shape, dtype=torch.float32))
        self.register_buffer("shifter_a", torch.zeros(mzi_property_shape, dtype=torch.float32))
        self.register_buffer("shifter_b", torch.ones(mzi_property_shape, dtype=torch.float32))
        self.register_buffer("shifter_c", torch.randn(mzi_property_shape, dtype=torch.float32))

        # 【修复1】：将 mesh_matrices 也注册为 buffer！让它归 PyTorch 管。
        mesh_shape = (self.in_blocks, self.out_blocks, self.parallel, self.parallel)
        self.register_buffer("mesh_matrices", torch.zeros(mesh_shape, dtype=torch.complex64))

        self.nonlinear = lambda x: x  # F.sigmoid

        # 初始化动作 (注意：此时依然在 CPU 上，等会会被 .to 转移)
        initial_action = torch.zeros(math.prod(param_shape), dtype=torch.float32)
        self.update_voltage(initial_action)

    def update_voltage(self, v_diff):
        """
        接收 PPO 给出的扁平化 action 向量，并更新所有的 8x8 MZI tile
        """
        # 【修复2】：强制将外部传进来的 v_diff 转换到模型所在的设备上！
        # 这一步极其重要，防止 PPO 传个 CPU 向量进来导致设备冲突。
        v_diff = v_diff.to(self.shifter_a.device)

        v_diff = v_diff.view(self.in_blocks, self.out_blocks, self.layer_num, self.parallel)

        updated_phase = (self.shifter_a * v_diff ** 2 +
                         self.shifter_b * v_diff +
                         self.shifter_c)
        self.global_phi.copy_(updated_phase)

        # 【修复3】：使用模型自己的 device 作为基准
        device = self.global_phi.device
        matrices = torch.zeros((self.in_blocks, self.out_blocks, self.parallel, self.parallel),
                               dtype=torch.complex64, device=device)

        for i in range(self.in_blocks):
            for j in range(self.out_blocks):
                # 【注意】：确保你写的 mzi_mesh 函数内部的张量也生成在了正确的 device 上！
                matrices[i, j] = mzi_mesh(self.global_phi[i, j], parallel=self.parallel)

        # 【修复4】：必须用 .copy_() 原地更新 buffer！绝对不能用 = 直接赋值！
        # 如果你写 self.mesh_matrices = matrices，刚刚注册的 buffer 追踪就断了。
        self.mesh_matrices.copy_(matrices)

    def forward(self, x):
        batch_size = x.size(0)

        if self.pad_in > 0:
            x = F.pad(x, (0, self.pad_in))

        x = x.to(dtype=torch.complex64)
        x_blocks = x.view(batch_size, self.in_blocks, self.parallel)

        out_blocks = torch.zeros((batch_size, self.out_blocks, self.parallel),
                                 dtype=torch.float32, device=x.device)

        for i in range(self.in_blocks):
            for j in range(self.out_blocks):
                mesh_ij = self.mesh_matrices[i, j]
                # 此时 x_blocks 和 mesh_ij 都在同一个 device (比如 GPU) 上，丝滑运算
                block_detected = (x_blocks[:, i, :] @ mesh_ij)
                block_detected = (block_detected.real.square() + block_detected.imag.square()).float()
                out_blocks[:, j, :] += block_detected

        out_flat = out_blocks.view(batch_size, -1)

        if self.pad_out > 0:
            out_flat = out_flat[:, :self.out_features]

        detected = out_flat.float()
        return self.nonlinear(detected)

    def step(self, x):
        return self.forward(x)

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

        # 【核心优化】：不再注册全零的协方差矩阵矩阵！
        # 论文方差是 0.04，标准差就是 sqrt(0.04) = 0.2
        # 我们只保留一个一维的标准差向量（内存占用从 1.5TB 降到几 KB）
        self.register_buffer('sigma', torch.full((act_space,), 0.2, dtype=torch.float32))

        self.loss_fn = torch.nn.functional.cross_entropy

    def configure_optimizers(self):
        # 优化器现在只优化 mu 这一个张量
        actor_optim = torch.optim.Adam([self.mu], lr=self.actor_lr)
        return actor_optim

    def get_action_and_logprob(self, action=None):
        # 【核心优化】：使用标准 Normal 分布代替复杂的 MultivariateNormal
        dist = Normal(self.mu, self.sigma)

        if action is None:
            action = dist.sample()

        # 【核心点】：各个物理参数是独立的，多元正态分布的对数概率
        # 等于每个独立一元正态分布对数概率的总和。
        # 如果 action 是单样本 [act_space]，则 sum(-1) 变成标量
        # 如果 action 是批次 [M_samples, act_space]，则 sum(-1) 变成 [M_samples]
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob
    def validation_step(self,batch_data, batch_idx):
        x_batch, y_target_batch = batch_data
        x_batch,y_target_batch=x_batch.cuda(),y_target_batch.cuda()
        # 【关键修复】：将图像 [batch_size, 1, 28, 28] 展平为 [batch_size, 784]
        x_batch = x_batch.view(x_batch.size(0), -1)
        y_theta_batch = self.env.step(x_batch)
        y_theta_batch=torch.argmax(y_theta_batch, dim=1)
        batch_acc=torch.sum(y_theta_batch==y_target_batch)/y_target_batch.size(0)
        self.log("val_acc", batch_acc.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
    def training_step(self, batch_data):
        # batch_data: 包含了一批输入信号 X 和对应的 Y_target [batch_size, parallel]
        x_batch, y_target_batch = batch_data
        x_batch,y_target_batch=x_batch.cuda(),y_target_batch.cuda()
        # 【关键修复】：将图像展平为 [batch_size, 784]
        x_batch = x_batch.view(x_batch.size(0), -1)
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

            # ✅ 工业级安全的防爆写法
            log_ratio = curr_log_probs - old_log_probs

            # 将对数差值限制在 [-20, 20] 之间。
            # torch.exp(20) 约为 4.8x10^8，足够策略进行剧烈优化，但绝对不会变成 inf 导致硬件溢出
            ratios = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))

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
