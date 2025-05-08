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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_mock_checkpoint(save_path='checkpoint_v6.pth', vocab_size=1000, embed_dim=512):
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists(save_path):
        mock_weights = torch.randn(vocab_size, embed_dim) * 0.02
        torch.save({'embed_weights': mock_weights}, save_path)

class MGAEmbedding(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=512, checkpoint_path='checkpoint_v6.pth'):
        super().__init__()
        generate_mock_checkpoint(checkpoint_path, vocab_size, embed_dim)
        pretrained = torch.load(checkpoint_path, map_location=device)
        self.base_embed = nn.Embedding.from_pretrained(pretrained['embed_weights'], freeze=False)
        
        self.projection = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, groups=embed_dim, padding=1),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, input_ids):
        embeddings = self.base_embed(input_ids)
        projected = self.projection(embeddings.permute(0,2,1)).permute(0,2,1)
        return F.normalize(projected, p=2, dim=-1)

class DynamicWeightMatrix(nn.Module):
    def __init__(self, num_attributes, init_factor=0.01, clip_threshold=0.7, checkpoint_path='checkpoint_v6.pth'):
        super().__init__()
        generate_mock_checkpoint(checkpoint_path, embed_dim=num_attributes)
        
        # 维度修正 (num_attributes,) -> (1, num_attributes)用于广播
        self.alpha = nn.Parameter(torch.randn(num_attributes) * init_factor)
        self.stability = nn.Parameter(torch.ones(num_attributes))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        pretrained = torch.load(checkpoint_path, map_location=device)
        self.register_buffer('historical_weights', pretrained['embed_weights'].mean(dim=0))
        
        self.clip_threshold = clip_threshold
        self.epsilon = 1e-6
        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.alpha.unsqueeze(0))
        nn.init.constant_(self.stability, 1.0)
        nn.init.constant_(self.temperature, 1.0)

    def forward(self, x, clip_features=None, epoch=None):
        # 维度对齐处理 (batch_size, num_attributes)
        adapted_x = x[:, :self.alpha.shape[0]].unsqueeze(-1)  # (B, A, 1)
        
        # 动态权重计算 (广播机制)
        scaled_logits = (self.alpha * adapted_x.squeeze(-1)) / self.temperature.clamp(min=0.1)
        raw_weights = self._stabilized_softmax(scaled_logits.unsqueeze(-1))  # (B, A, 1)
        
        # 历史权重融合 (增加batch维度)
        decay_factor = 1 / (1 + (epoch/10 if epoch else 0))
        stability = self.stability.unsqueeze(0)  # (1, A)
        historical = self.historical_weights.unsqueeze(0).to(x.device)  # (1, A)
        
        weighted_weights = decay_factor * stability * raw_weights.squeeze(-1) + \
                         (1 - decay_factor) * historical

        # CLIP语义约束
        if clip_features is not None:
            clip_tensor = clip_features.to(x.device).unsqueeze(1)  # (B, 1, D)
            similarity = F.cosine_similarity(weighted_weights.unsqueeze(1), clip_tensor, dim=-1)
            mask = (similarity > self.clip_threshold).float().detach()
            weighted_weights = weighted_weights * (1 - mask) + historical * mask

        return self._output_handler(weighted_weights)

    def _stabilized_softmax(self, logits):
        safe_logits = logits - logits.max(dim=1, keepdim=True).values
        return F.softmax(safe_logits + self.epsilon, dim=1)

    def _output_handler(self, weights):
        return weights if self.training else weights.detach()

def validate_dynamic_matrix():
    generate_mock_checkpoint()
    model = DynamicWeightMatrix(512).to(device)
    test_input = torch.randn(8, 512).to(device)
    
    # 基础验证
    weights = model(test_input)
    assert weights.shape == (8, 512), f"维度错误: 获得{weights.shape}, 预期(8,512)"
    assert torch.allclose(weights.sum(dim=1), torch.ones(8).to(device), atol=1e-4), "概率分布异常"
    
    # CLIP约束测试
    clip_model, _ = clip.load("ViT-B/32", device=device)
    text_features = clip_model.encode_text(clip.tokenize(["cyberpunk"])).float().to(device)
    constrained_weights = model(test_input, text_features)
    similarity = F.cosine_similarity(constrained_weights, text_features.unsqueeze(1), dim=-1)
    assert (similarity < 0.7).all(), f"CLIP约束失效，最大相似度{similarity.max().item():.4f}"
    
    print("动态权重矩阵验证通过")

if __name__ == "__main__":
    validate_dynamic_matrix()

