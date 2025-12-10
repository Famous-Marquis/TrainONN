import torch
from pytorch_lightning.loggers import mlflow
from torch import nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# --- 1. 基础组件 (物理算子模拟) ---

class PhysicalMatrix(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 使用 nn.Parameter 使得模型保存时能包含此权重
        self.weight_matrix = nn.Parameter(torch.randn((in_features, out_features)) * 0.01)
        self.generator = torch.Generator()
        self.generator.manual_seed(42)  # 固定种子以便复现

    def forward(self, x):
        # x: [Batch, In] @ W: [In, Out] -> [Batch, Out]
        return x @ self.weight_matrix

    def backward(self, d_z):
        """
        物理反向传播模拟
        d_z: 下一层的梯度 [Batch, Out]
        return: d_x [Batch, In]
        """
        # 模拟物理硬件中的非对称噪声 (Asymmetric Noise)
        # 注意：此处必须使用 .detach() 避免 PyTorch 记录此处的计算图，因为我们是手动微分
        noise = 0.00 * torch.randn(self.weight_matrix.shape, generator=self.generator).to(d_z.device)
        noisy_weight_T = (self.weight_matrix + noise).T

        # d_z: [Batch, Out] @ W_noisy.T: [Out, In] -> [Batch, In]
        return d_z @ noisy_weight_T


class PhysicalActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

    def backward(self, x):
        # ReLU 导数: x > 0 为 1, 否则为 0
        return (x > 0).float()


class PhysicalLoss:
    """如果不含参数，可以不继承 nn.Module"""

    def forward(self, pred, target):
        return F.mse_loss(pred, target)

    def backward(self, pred, target):
        """
        MSE 导数: dL/d_pred = 2/N * (pred - target)
        这里返回 Tensor 形状 [Batch, Out_features]
        """
        batch_size = pred.shape[0]
        # 注意维度保持，以便进行矩阵乘法
        return 2. * (pred - target) / batch_size


# --- 2. 物理层 (管理前向与反向状态) ---

class PhysicalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.matrix = PhysicalMatrix(in_features, out_features)
        self.activation = PhysicalActivation()

        # 暂存物理状态 (相当于电容或暂存器)
        self.input_cache = None
        self.pre_act_cache = None

    def forward(self, x):
        self.input_cache = x.detach()  # 记录输入物理量
        z = self.matrix.forward(x)
        self.pre_act_cache = z.detach()  # 记录激活前物理量
        a = self.activation.forward(z)
        return a

    def backward(self, d_a):
        """
        显式计算梯度
        :param d_a: 上一级的梯度 [Batch, Out]
        :return: d_x (传给前一层), d_w (当前层权重更新量)
        """
        # 1. 激活函数反向
        # d_loss/d_z = d_loss/d_a * d_a/d_z
        d_act = self.activation.backward(self.pre_act_cache)
        d_z = d_a * d_act

        # 2. 权重梯度计算 (物理测量)
        # d_w = X.T @ d_z
        # [Batch, In].T @ [Batch, Out] -> [In, Out]
        # 必须处理维度问题
        d_w = self.input_cache.T @ d_z

        # 3. 输入梯度计算 (传递给上一层)
        d_x = self.matrix.backward(d_z)

        # 清空状态以节省显存/模拟放电
        self.input_cache = None
        self.pre_act_cache = None

        return d_x, d_w


# --- 3. Lightning Module (控制器) ---

class PhysicalNetworkSystem(L.LightningModule):
    def __init__(self, in_features, out_features, lr=0.01):
        super().__init__()
        self.save_hyperparameters()

        # 定义网络结构
        self.layer1 = PhysicalLayer(in_features, out_features)
        self.loss_fn = PhysicalLoss()

        # 重要：关闭自动优化，因为我们要手动应用 d_w
        self.automatic_optimization = False

    def forward(self, x):
        return self.layer1.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # --- 1. 前向过程 (Forward Pass) ---
        y_pred = self(x)

        # --- 2. 显式计算 Loss ---
        loss = self.loss_fn.forward(y_pred, y)

        # --- 3. 显式物理反向传播 (Explicit Backward) ---
        # 计算 Loss 对 输出的梯度
        d_loss_d_pred = self.loss_fn.backward(y_pred, y)

        # 获得物理梯度 dx 和 dw
        _, d_w = self.layer1.backward(d_loss_d_pred)

        # --- 4. 手动优化 (Optimization Step) ---
        # 获取我们定义的优化器 (虽是手动，但利用 Lightning 管理参数是个好习惯，或者直接手写更新)
        # 这里演示纯手写更新，模拟物理硬件调节
        lr = self.hparams.lr
        with torch.no_grad():
            self.layer1.matrix.weight_matrix -= lr * d_w

        # --- 5. Logging ---
        # prog_bar=True 会在进度条显示
        self.log("train_loss", loss, prog_bar=True)
        # 记录梯度的范数，观察物理噪声的影响
        self.log("grad_norm", d_w.norm(), prog_bar=False)

        return loss

    def configure_optimizers(self):
        # 因为我们是手动更新权重 (line 124)，这里返回 None
        # 如果你想用 Adam 等标准优化器来应用你的 d_w，这里可以返回 standard optimizer
        return None
class DigitalNetworkSystem(L.LightningModule):
    def __init__(self, in_features, out_features, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.layer_=nn.Sequential(nn.Linear(in_features, out_features),nn.Softmax(dim=1))
        self.loss_fn=nn.CrossEntropyLoss()
    def forward(self, x):
        return self.layer_(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn.forward(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
# --- 4. 运行实验 ---

if __name__ == "__main__":
    # 1. 准备假数据
    in_dim, out_dim = 10, 4
    N_samples = 1000
    X = torch.randn(N_samples, in_dim)
    # 目标是学习一个简单的线性映射 Y = XW + b
    # true_W=torch.tensor([])
    true_W = torch.randn(in_dim, out_dim)
    Y = X @ true_W + 0.0 * torch.randn(N_samples, out_dim)

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    # 2. 初始化系统
    model = PhysicalNetworkSystem(in_features=in_dim, out_features=out_dim, lr=1e-4)
    model_digital=DigitalNetworkSystem(in_features=in_dim, out_features=out_dim, lr=1e-4)
    # 3. 配置 Trainer
    # max_epochs: 训练轮数
    # log_every_n_steps: 多少步记录一次 log
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="gpu",  # 如果有 GPU 改为 "gpu"
        log_every_n_steps=10,
        enable_checkpointing=False,
    )

    # 4. 开始训练
    print("开始物理神经网络训练模拟...")
    # trainer.fit(model, dataloader,val_dataloaders=val_dataloader,)
    print("训练结束.")

    trainer.fit(model_digital,dataloader)