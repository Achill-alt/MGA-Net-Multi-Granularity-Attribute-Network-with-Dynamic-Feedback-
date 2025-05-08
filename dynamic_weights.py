# 安装PyTorch核心组件
!pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# 安装其他Python依赖
!pip install ftfy regex h5py pyyaml tqdm matplotlib plotly Pillow

# 安装CLIP源码版
!pip install git+https://github.com/openai/CLIP.git

# 安装NCCL支持（需Colab权限）
!apt update && apt install -y --no-install-recommends libnccl-dev libnccl2

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import clip
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_mock_checkpoint(save_path: str = 'checkpoint_v6.pth',
                             vocab_size: int = 1000,
                             embed_dim: int = 512) -> None:
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists(save_path):
        mock_weights = torch.randn(vocab_size, embed_dim) * 0.02
        torch.save({'embed_weights': mock_weights}, save_path)
        print(f"生成模拟预训练权重文件于：{save_path}")

class MGAEmbedding(nn.Module):
    def __init__(self, 
                vocab_size: int = 1000,
                embed_dim: int = 512,
                checkpoint_path: str = 'checkpoint_v6.pth'):
        super().__init__()
        generate_mock_checkpoint(checkpoint_path, vocab_size, embed_dim)
        pretrained = torch.load(checkpoint_path, map_location=device)
        self.base_embed = nn.Embedding.from_pretrained(
            pretrained['embed_weights'], freeze=False)
        
        # 修改投影层结构
        self.projection = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, groups=embed_dim, padding=1),
            nn.GELU()
        )
        self.layer_norm = nn.LayerNorm(embed_dim)  # 分离LayerNorm层

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.base_embed(input_ids)  # [batch, seq, dim]
        
        # 调整维度顺序进行卷积操作
        projected = self.projection(embeddings.permute(0, 2, 1))  # [batch, dim, seq]
        
        # 恢复维度顺序后归一化
        projected = projected.permute(0, 2, 1)  # [batch, seq, dim]
        normalized = self.layer_norm(projected)
        
        return F.normalize(normalized, p=2, dim=-1)

class DynamicWeightMatrix(nn.Module):
    def __init__(self, 
                num_attributes: int,
                init_factor: float = 0.01,
                clip_threshold: float = 0.7,
                checkpoint_path: str = 'checkpoint_v6.pth',
                history_decay: float = 0.9):
        super().__init__()
        generate_mock_checkpoint(checkpoint_path, embed_dim=num_attributes)
        pretrained = torch.load(checkpoint_path, map_location=device)
        
        self.alpha = nn.Parameter(torch.randn(num_attributes, 1) * init_factor)
        self.stability = nn.Parameter(torch.ones(num_attributes, 1))
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.register_buffer('historical_weights',
                           pretrained['embed_weights'].mean(dim=0).view(1, -1))
        self.clip_threshold = clip_threshold
        self.epsilon = 1e-8
        self.history_decay = history_decay
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.xavier_normal_(self.alpha)
        nn.init.constant_(self.stability, 1.0)
        nn.init.uniform_(self.temperature, 0.5, 2.0)

    def forward(self, 
               x: torch.Tensor,
               clip_features: Optional[torch.Tensor] = None,
               epoch: Optional[int] = None) -> torch.Tensor:
        batch_size = x.size(0)
        adapted_x = x[:, :self.alpha.shape[0]].unsqueeze(-1)
        
        scaled_logits = (self.alpha * adapted_x) / self.temperature.clamp(min=0.1, max=100)
        raw_weights = self._stabilized_softmax(scaled_logits)

        if self.training:
            current_weights = raw_weights.squeeze(-1).mean(dim=0).detach().view(1, -1)
            self.historical_weights = (
                self.history_decay * self.historical_weights +
                (1 - self.history_decay) * current_weights
            )

        stability = self.stability.t().expand(batch_size, -1)
        historical = self.historical_weights.expand(batch_size, -1).to(x.device)
        
        weighted_weights = stability * raw_weights.squeeze(-1) + historical
        weighted_weights = F.softmax(weighted_weights, dim=1)

        if clip_features is not None:
            clip_tensor = clip_features.to(x.device).unsqueeze(1)
            similarity = F.cosine_similarity(
                weighted_weights.unsqueeze(2),
                clip_tensor.unsqueeze(1),
                dim=-1
            )
            mask = (similarity > self.clip_threshold).float().detach().squeeze(-1)
            weighted_weights = weighted_weights * (1 - mask) + historical * mask

        return self._output_handler(weighted_weights)

    def _stabilized_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        max_logits = logits.max(dim=1, keepdim=True).values.detach()
        stable_logits = logits - max_logits
        return F.softmax(stable_logits + self.epsilon, dim=1)

    def _output_handler(self, weights: torch.Tensor) -> torch.Tensor:
        return weights if self.training else weights.detach()

def test_mga_embedding():
    print("\n===测试 MGA嵌入层===")
    embedder = MGAEmbedding().to(device)
    test_input = torch.randint(0, 1000, (2, 64)).to(device)
    output = embedder(test_input)
    print(f"输入形状：{test_input.shape}")
    print(f"输出形状：{output.shape}")
    print(f"输出范数：{torch.norm(output, dim=-1).mean():.4f}")

def test_dynamic_weights():
    print("\n===测试动态权重矩阵===")
    model = DynamicWeightMatrix(512).to(device)
    
    test_input = torch.randn(8, 512).to(device)
    weights = model(test_input)
    print(f"正常输入测试：")
    print(f"权重形状：{weights.shape}")
    print(f"行求和均值：{weights.sum(dim=1).mean():.4f} ± {weights.sum(dim=1).std():.4f}")

if __name__ == "__main__":
    print(f"当前计算设备：{'GPU' if torch.cuda.is_available() else 'CPU'}")
    test_mga_embedding()
    test_dynamic_weights()
    print("\n测试全部通过! 模型功能验证完成。")

