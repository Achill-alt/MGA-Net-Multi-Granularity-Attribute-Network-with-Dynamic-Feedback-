# 必需依赖项安装命令(Google Colab环境)
# 基础依赖
!pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
!pip install numpy>=1.20.0

# CLIP模型支持
!pip install ftfy regex
!pip install git+https://github.com/openai/CLIP.git

# 分布式训练支持(NCCL后端)
!apt install -y --no-install-recommends libnccl-dev libnccl2

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

# =====================================================================================================
# 属性嵌入层实现(维度修正)
class MGAEmbedding(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=512, reduced_dim=512):
        super().__init__()
        # 模拟预训练权重加载(实际使用时替换为checkpoint_v6.pth)
        pretrained_weights = torch.randn(vocab_size, embed_dim) * 0.02
        self.base_embed = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
        self.projection = nn.Identity()  # 保持原始512维

    def forward(self, input_ids):
        embeddings = self.base_embed(input_ids)
        return F.normalize(embeddings, p=2, dim=-1)

# =====================================================================================================
# 动态权重层实现(维度适配修正)
class DynamicWeightLayer(nn.Module):
    def __init__(self, num_attributes, init_beta=0.7):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(num_attributes) * 0.01)  # 缩小初始化范围
        self.beta = nn.Parameter(torch.ones(num_attributes) * init_beta)
        
    def forward(self, x, epoch):
        adapted_x = x[:, :self.alpha.shape[0]]  # 截取前num_attributes个维度
        raw_weights = F.softmax(self.alpha * adapted_x, dim=-1)
        decay_factor = 1 / (1 + epoch/10)
        weighted = decay_factor * self.beta * raw_weights
        return weighted  # 直接返回权重矩阵

# =====================================================================================================
# 测试验证模块(修正维度验证)
def run_unit_tests():
    # 测试嵌入层
    embed_layer = MGAEmbedding()
    dummy_input = torch.randint(0, 1000, (32, 10))
    output = embed_layer(dummy_input)
    print(f"Embedding Test: {output.shape} (应输出 torch.Size([32, 10, 512]))")
    
    # 测试权重层
    weight_layer = DynamicWeightLayer(num_attributes=512)  # 匹配嵌入维度
    test_input = torch.randn(32, 512)
    weights = weight_layer(test_input, epoch=5)
    print(f"\nWeight Layer Test: 权重范围[{weights.min():.4f}, {weights.max():.4f}] (应介于0-1之间)")

# =====================================================================================================
# CLIP兼容性检查(维度匹配修正)
def clip_compatibility_check(model, text_prompts, device="cuda"):
    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 生成实际输入(适配512维特征)
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)
    
    with torch.no_grad():
        embeddings = model(input_ids)
        text_features = clip_model.encode_text(clip.tokenize(text_prompts).to(device))
    
    # 维度对齐处理(调整序列维度)
    emb_features = embeddings.mean(dim=1)  # [batch, seq, dim]->[batch, dim]
    text_features = text_features.float()  # 确保精度一致
    
    # 计算相似度(修正维度对齐)
    similarity = F.cosine_similarity(emb_features, text_features, dim=-1)
    conflict_mask = (similarity < 0.7) & (text_features.norm(dim=-1) > 0.8)
    
    if conflict_mask.any():
        print(f"检测到{conflict_mask.sum()}个冲突特征，已冻结梯度")
        embeddings = embeddings.detach()
    
    return embeddings

# =====================================================================================================
if __name__ == "__main__":
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")
    
    # 初始化模型
    embed_model = MGAEmbedding().to(device)
    weight_layer = DynamicWeightLayer(num_attributes=512).to(device)  # 匹配嵌入维度
    
    # 运行单元测试
    print("\n=== 开始单元测试 ===")
    run_unit_tests()
    
    # CLIP兼容性检查示例
    print("\n=== CLIP兼容性检查 ===")
    test_prompts = ["a photo of a cat", "a cyberpunk cityscape"]
    embeddings = clip_compatibility_check(embed_model, test_prompts, device)
    print(f"输出嵌入形状: {embeddings.shape}")
    
    # 训练示例(添加梯度裁剪)
    print("\n=== 训练演示 ===")
    optimizer = torch.optim.Adam([
        {'params': embed_model.parameters()},
        {'params': weight_layer.parameters(), 'weight_decay': 0.01}
    ], lr=1e-3)
    
    # 虚拟训练数据(适配512维)
    dummy_data = torch.randint(0, 1000, (64, 10)).to(device)
    
    for epoch in range(3):
        optimizer.zero_grad()
        
        # 前向传播
        embeds = embed_model(dummy_data)
        weights = weight_layer(embeds.mean(dim=1), epoch)
        
        # 模拟损失函数：权重正则化 + 特征保留约束
        loss = weights.mean() + 0.1 * embeds.pow(2).mean()  # 添加正则项防止负损失
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(embed_model.parameters(), 1.0)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")