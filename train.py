import mlflow
import torch
from lightning import Trainer
from mlflow import MlflowClient
import lightning as L
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from model import PPO, DNN, mzi_mesh, SimMZIMitrix


def get_simple_data(n_samples=1000,n_features=8):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,  # 总维度：8个特征
        n_informative=5,  # 有效特征：其中5个维度包含对分类有用的信息
        n_redundant=2,  # 冗余特征：其中2个是由有效特征线性组合产生的（模拟多重共线性）
        n_repeated=0,  # 重复特征：0个
        n_classes=2,  # 类别数量：2个分类 (0 和 1)
        weights=[0.5, 0.5],  # 类别均衡：正负样本各占 50%
        random_state=42  # 随机种子，确保每次生成的数据一致，方便复现和调试
    )
    X, y = torch.tensor(X, dtype=torch.float32).cuda(), torch.tensor(y, dtype=torch.long).cuda()
    y = one_hot(y, num_classes=2)
    y = torch.nn.functional.pad(y, (0, 6), "constant", 0)
    assert X.shape == y.shape
    return X, y


if __name__ == '__main__':
    trainer=Trainer()
    client = MlflowClient()
    N = 1024
    size = 8
    X,y=get_simple_data(n_samples=N, n_features=size)
    train_dataset = TensorDataset(X,y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
    for batch in train_loader:
        X,y=batch
    # mzi = mzi_mesh(np.random.randn(10, 8), parallel=8)
    mzi = SimMZIMitrix(10, 8)
    ppo_agent = PPO(DNN, batch=None, act_space=80, obs_space=16, actor_lr=1e-3, critic_lr=1e-3, mzi=mzi,
                    epsilon=0.02).cuda()
    trainer.fit(ppo_agent, train_loader)

