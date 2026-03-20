from abc import ABC, abstractmethod

import numpy as np
import torch


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

class PPO:
    def __init__(self,NN,act_space,obs_space,actor_lr,critic_lr,mzi):
        """

        :param act_space: tap to control [layer_num * parallel]
        :param obs_space: MZI inp and oup [parallel*2,] X cat Y_theta
        """
        self.env=mzi
        self.actor=NN(obs_space,act_space)
        self.critic=NN(obs_space,1)

        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)



        ...
if __name__ == '__main__':
    symmetric_mzi_matrix(np.ones((6,)))
    matrix_mesh = mzi_mesh(np.ones((10, 8)))
    print(matrix_mesh.shape)
