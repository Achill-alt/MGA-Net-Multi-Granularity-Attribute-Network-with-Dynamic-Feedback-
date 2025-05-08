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
import clip
import os

class MGAEmbedding(nn.Module):
    """属性嵌入层(集成历史权重加载机制)"""
    def __init__(self, vocab_size=1000, embed_dim=512, checkpoint_path='checkpoint_v6.pth'):
        super().__init__()
        self._load_pretrained_weights(checkpoint_path, vocab_size, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

    def _load_pretrained_weights(self, path, vocab_size, embed_dim):
        if not os.path.exists(path):
            pretrained = torch.randn(vocab_size, embed_dim) * 0.02
            torch.save({'embed_weights': pretrained}, path)
        state_dict = torch.load(path, map_location='cpu')
        self.base_embed = nn.Embedding.from_pretrained(
            state_dict['embed_weights'],
            freeze=False
        )

    def forward(self, input_ids):
        embeddings = self.base_embed(input_ids)
        projected = self.projection(embeddings)
        return F.normalize(projected, p=2, dim=-1)

class DynamicWeightMatrix(nn.Module):
    """动态权重分配矩阵(集成历史反馈机制)"""
    def __init__(self, num_attributes=256, history_decay=0.7, clip_threshold=0.7):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(num_attributes) * 0.01)
        self.temperature = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer('historical_weights', 
                           torch.ones(num_attributes) / num_attributes)
        self.history_decay = history_decay
        self.clip_threshold = clip_threshold
        self.epsilon = 1e-6
        self.clip_proj = nn.Linear(512, num_attributes)

    def _stabilized_softmax(self, logits):
        max_logits = logits.max(dim=-1, keepdim=True).values.detach()
        stable_logits = logits - max_logits
        return F.softmax(stable_logits / self.temperature + self.epsilon, dim=-1)

    def _apply_clip_constraints(self, weights, clip_features):
        if clip_features is None:
            return weights
        
        clip_projected = self.clip_proj(clip_features)
        similarity = F.cosine_similarity(
            weights.unsqueeze(1),
            clip_projected.unsqueeze(0),
            dim=-1
        )
        max_similarity, _ = similarity.max(dim=1, keepdim=True)
        mask = (max_similarity > self.clip_threshold).float()
        return weights * (1 - mask) + self.historical_weights * mask

    def forward(self, x, clip_features=None, update_history=True):
        # 修正：移除多余的均值计算，直接使用投影后的特征
        logits = self.alpha * x  
        raw_weights = self._stabilized_softmax(logits)
        
        if update_history and self.training:
            current_weights = raw_weights.mean(dim=0).detach()
            self.historical_weights = (self.history_decay * self.historical_weights + 
                                      (1 - self.history_decay) * current_weights)
        
        weighted = self._apply_clip_constraints(raw_weights, clip_features)
        return F.normalize(weighted, p=1, dim=-1)

class MGANetwork(nn.Module):
    """完整MGA网络架构"""
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.embedding = MGAEmbedding().to(device)
        self.weight_matrix = DynamicWeightMatrix().to(device)
        self._init_clip_model()

    def _init_clip_model(self):
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, text_prompts=None):
        embeddings = self.embedding(input_ids)
        clip_features = None
        
        if text_prompts is not None:
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            with torch.no_grad():
                clip_features = self.clip_model.encode_text(text_tokens).float()
        
        # 修正：使用正确的维度计算均值
        weights = self.weight_matrix(embeddings.mean(dim=1), clip_features)
        weighted_features = torch.einsum('bsd,bd->bsd', embeddings, weights)
        return weighted_features

def validate_integration():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"验证设备: {device.upper()}")
    
    model = MGANetwork(device=device)
    test_input = torch.randint(0, 1000, (4, 16)).to(device)
    
    output = model(test_input, ["cyberpunk city"] * 4)
    
    assert output.shape == (4, 16, 256), "输出形状错误"
    weight_sum = model.weight_matrix.historical_weights.sum().item()
    assert abs(weight_sum - 1.0) < 0.01, "历史权重未归一化"
    
    print("\n验证结果:")
    print(f"- 输出形状: {tuple(output.shape)}")
    print(f"- 历史权重合计: {weight_sum:.4f}")
    print("- 所有测试通过!")

if __name__ == "__main__":
    validate_integration()
