import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch import nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np


# --- 1. 物理组件 (保持不变，增加灵活性) ---

class PhysicalLayer(nn.Module):
    def __init__(self, in_features, out_features, noise_scale=0.05, activation='relu'):
        super().__init__()
        # 初始化权重：物理系统通常需要较小的初始值
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.noise_scale = noise_scale
        self.activation_type = activation
        self.generator = torch.Generator().manual_seed(42)

        # 物理状态缓存
        self.input_cache = None
        self.z_cache = None

    def forward(self, x):
        self.input_cache = x.detach()
        z = x @ self.weight + self.bias
        self.z_cache = z.detach()

        if self.activation_type == 'relu':
            return F.relu(z)
        elif self.activation_type == 'identity':
            return z  # 最后一层通常不需要激活，或者由外部Softmax处理
        return z

    def backward(self, d_a):
        # 1. 激活函数反向
        if self.activation_type == 'relu':
            d_act = (self.z_cache > 0).float()
            d_z = d_a * d_act
        else:  # identity
            d_z = d_a

        # 2. 物理噪声模拟 (权重读取噪声)
        noise = self.noise_scale * torch.randn(self.weight.shape,
                                               generator=self.generator).to(d_z.device)
        noisy_weight_T = (self.weight + noise).T

        # 3. 反向传播
        d_x = d_z @ noisy_weight_T
        d_w = self.input_cache.T @ d_z
        # Z= W @ X + b <=> W @ X + b @ I (单位阵)
        d_b = d_z.sum(dim=0)

        self.input_cache = None
        self.z_cache = None

        return d_x, {"weight": d_w, "bias": d_b}


class PhysicalMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        # 构建层：除了最后一层，都用 ReLU
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            act = 'identity' if is_last else 'relu'
            self.layers.append(PhysicalLayer(layer_sizes[i],
                                             layer_sizes[i + 1],
                                             activation=act))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def manual_backward_pass(self, d_loss_final):
        grads_all = []
        d_current = d_loss_final
        for layer in reversed(self.layers):
            d_current, grads = layer.backward(d_current)
            grads_all.insert(0, grads)
        return grads_all


# --- 2. Lightning 系统 (添加准确率监控) ---

class IrisPhysicalSystem(L.LightningModule):
    def __init__(self, layer_sizes, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model = PhysicalMLP(layer_sizes)
        self.automatic_optimization = False  # 手动优化

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch  # y_true 是 One-Hot 编码的 [Batch, 3]

        # 1. Forward
        y_pred = self(x)

        # 使用 MSE Loss (适合物理测量)
        loss = F.mse_loss(y_pred, y_true)

        # 2. Backward (手动物理反向)
        # MSE Gradient: 2/N * (y_pred - y_true)
        d_loss = 2. * (y_pred - y_true) / x.size(0)
        grads = self.model.manual_backward_pass(d_loss)

        # 3. Update (模拟物理调节)
        lr = self.hparams.lr
        with torch.no_grad():
            for i, layer in enumerate(self.model.layers):
                layer.weight -= lr * grads[i]['weight']
                layer.bias -= lr * grads[i]['bias']

        # 计算准确率 (仅用于观察，不参与梯度)
        acc = (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)  # 进度条显示准确率
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y_true)
        acc = (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return None


# --- 3. 数据准备 (Iris Dataset) ---

def get_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # 1. 归一化 (Scale) - 对物理网络至关重要
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. One-Hot 编码 (Label -> Vector)
    # 0 -> [1, 0, 0], 1 -> [0, 1, 0]
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    # 3. 转换为 Tensor
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_onehot, dtype=torch.float32)

    # 4. 分割数据集
    X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    return DataLoader(train_ds, batch_size=16, shuffle=True), DataLoader(val_ds, batch_size=16)


# --- 4. 运行实验 ---

if __name__ == "__main__":
    train_loader, val_loader = get_iris_data()

    # Iris: 4个特征 -> 隐藏层 -> 3个类别
    # 结构: [4, 16, 3]
    system = IrisPhysicalSystem(layer_sizes=[4, 16, 3], lr=0.05)
    mllogger=MLFlowLogger(experiment_name="Iris_PhysicalSystem",
                          run_name="test",log_model=False,tracking_uri="file:./mlruns")

    trainer = L.Trainer(
        max_epochs=50,  # 训练 50 轮
        accelerator="gpu",
        log_every_n_steps=1,  # 这里的步数很少，每步都记
        logger=mllogger,
        enable_checkpointing=False
    )

    print("--- 开始 Iris 物理训练 ---")
    trainer.fit(system, train_loader, val_loader)