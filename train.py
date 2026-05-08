import mlflow
import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
import lightning as L
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from model import DNN, mzi_mesh, SimMZIMitrix, PPOFixedMZI, SimpleModel


def get_simple_data(n_samples=1000,n_features=8):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,  # 总维度：8个特征
        n_informative=8,  # 有效特征：其中5个维度包含对分类有用的信息
        n_redundant=0,  # 冗余特征：其中2个是由有效特征线性组合产生的（模拟多重共线性）
        n_repeated=0,  # 重复特征：0个
        n_classes=8,  # 类别数量：2个分类 (0 和 1)
        weights=None,  # 类别均衡：正负样本各占 50%
        random_state=42  # 随机种子，确保每次生成的数据一致，方便复现和调试
    )
    X, y = torch.tensor(X, dtype=torch.float32).cuda(), torch.tensor(y, dtype=torch.long).cuda()
    # y = one_hot(y, num_classes=8)
    # y = torch.nn.functional.pad(y, (0, 6), "constant", 0)

    return X, y


if __name__ == '__main__':

    logger=MLFlowLogger(tracking_uri='file:./mlruns',
                        experiment_name = "PPOFixedMZI",
                        run_name="CE 指给"
    )
    trainer=Trainer(max_epochs=200,logger=logger,check_val_every_n_epoch=1)
    N = 40960
    size = 8
    X,y=get_simple_data(n_samples=N, n_features=size)
    train_dataset = TensorDataset(X,y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    X,y=get_simple_data(n_samples=1024, n_features=size)
    val_dataset = TensorDataset(X,y)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

    # mzi = mzi_mesh(np.random.randn(10, 8), parallel=8)
    mzi = SimMZIMitrix(10, 8)
    ppo_agent = PPOFixedMZI(M_samples=16, act_space=80,  actor_lr=1e-2, mzi=mzi,
                    epsilon=0.2).cuda()
    simple_nn=SimpleModel()
    trainer.fit(ppo_agent, train_loader,val_loader)

