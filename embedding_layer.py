# ���������װ����(Google Colab����)
# ��������
!pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
!pip install numpy>=1.20.0

# CLIPģ��֧��
!pip install ftfy regex
!pip install git+https://github.com/openai/CLIP.git

# �ֲ�ʽѵ��֧��(NCCL���)
!apt install -y --no-install-recommends libnccl-dev libnccl2

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

# =====================================================================================================
# ����Ƕ���ʵ��(ά������)
class MGAEmbedding(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=512, reduced_dim=512):
        super().__init__()
        # ģ��Ԥѵ��Ȩ�ؼ���(ʵ��ʹ��ʱ�滻Ϊcheckpoint_v6.pth)
        pretrained_weights = torch.randn(vocab_size, embed_dim) * 0.02
        self.base_embed = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
        self.projection = nn.Identity()  # ����ԭʼ512ά

    def forward(self, input_ids):
        embeddings = self.base_embed(input_ids)
        return F.normalize(embeddings, p=2, dim=-1)

# =====================================================================================================
# ��̬Ȩ�ز�ʵ��(ά����������)
class DynamicWeightLayer(nn.Module):
    def __init__(self, num_attributes, init_beta=0.7):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(num_attributes) * 0.01)  # ��С��ʼ����Χ
        self.beta = nn.Parameter(torch.ones(num_attributes) * init_beta)
        
    def forward(self, x, epoch):
        adapted_x = x[:, :self.alpha.shape[0]]  # ��ȡǰnum_attributes��ά��
        raw_weights = F.softmax(self.alpha * adapted_x, dim=-1)
        decay_factor = 1 / (1 + epoch/10)
        weighted = decay_factor * self.beta * raw_weights
        return weighted  # ֱ�ӷ���Ȩ�ؾ���

# =====================================================================================================
# ������֤ģ��(����ά����֤)
def run_unit_tests():
    # ����Ƕ���
    embed_layer = MGAEmbedding()
    dummy_input = torch.randint(0, 1000, (32, 10))
    output = embed_layer(dummy_input)
    print(f"Embedding Test: {output.shape} (Ӧ��� torch.Size([32, 10, 512]))")
    
    # ����Ȩ�ز�
    weight_layer = DynamicWeightLayer(num_attributes=512)  # ƥ��Ƕ��ά��
    test_input = torch.randn(32, 512)
    weights = weight_layer(test_input, epoch=5)
    print(f"\nWeight Layer Test: Ȩ�ط�Χ[{weights.min():.4f}, {weights.max():.4f}] (Ӧ����0-1֮��)")

# =====================================================================================================
# CLIP�����Լ��(ά��ƥ������)
def clip_compatibility_check(model, text_prompts, device="cuda"):
    # ����CLIPģ��
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # ����ʵ������(����512ά����)
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)
    
    with torch.no_grad():
        embeddings = model(input_ids)
        text_features = clip_model.encode_text(clip.tokenize(text_prompts).to(device))
    
    # ά�ȶ��봦��(��������ά��)
    emb_features = embeddings.mean(dim=1)  # [batch, seq, dim]->[batch, dim]
    text_features = text_features.float()  # ȷ������һ��
    
    # �������ƶ�(����ά�ȶ���)
    similarity = F.cosine_similarity(emb_features, text_features, dim=-1)
    conflict_mask = (similarity < 0.7) & (text_features.norm(dim=-1) > 0.8)
    
    if conflict_mask.any():
        print(f"��⵽{conflict_mask.sum()}����ͻ�������Ѷ����ݶ�")
        embeddings = embeddings.detach()
    
    return embeddings

# =====================================================================================================
if __name__ == "__main__":
    # �����豸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"�����豸: {device}")
    
    # ��ʼ��ģ��
    embed_model = MGAEmbedding().to(device)
    weight_layer = DynamicWeightLayer(num_attributes=512).to(device)  # ƥ��Ƕ��ά��
    
    # ���е�Ԫ����
    print("\n=== ��ʼ��Ԫ���� ===")
    run_unit_tests()
    
    # CLIP�����Լ��ʾ��
    print("\n=== CLIP�����Լ�� ===")
    test_prompts = ["a photo of a cat", "a cyberpunk cityscape"]
    embeddings = clip_compatibility_check(embed_model, test_prompts, device)
    print(f"���Ƕ����״: {embeddings.shape}")
    
    # ѵ��ʾ��(����ݶȲü�)
    print("\n=== ѵ����ʾ ===")
    optimizer = torch.optim.Adam([
        {'params': embed_model.parameters()},
        {'params': weight_layer.parameters(), 'weight_decay': 0.01}
    ], lr=1e-3)
    
    # ����ѵ������(����512ά)
    dummy_data = torch.randint(0, 1000, (64, 10)).to(device)
    
    for epoch in range(3):
        optimizer.zero_grad()
        
        # ǰ�򴫲�
        embeds = embed_model(dummy_data)
        weights = weight_layer(embeds.mean(dim=1), epoch)
        
        # ģ����ʧ������Ȩ������ + ��������Լ��
        loss = weights.mean() + 0.1 * embeds.pow(2).mean()  # ����������ֹ����ʧ
        
        # �ݶȲü�
        torch.nn.utils.clip_grad_norm_(embed_model.parameters(), 1.0)
        
        # ���򴫲�
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")