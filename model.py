import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================= Spectral Module =======================
class SpectralGatingUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.filter_generator = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim * 2, 1)
        )

    def forward(self, x):
        B, C, L = x.shape
        x_fft = torch.fft.rfft(x, dim=2)

        weights = self.filter_generator(x.mean(dim=-1, keepdim=True)).view(B, C, 2)
        real_w, imag_w = weights[..., 0], weights[..., 1]

        real = x_fft.real * real_w.unsqueeze(-1) - x_fft.imag * imag_w.unsqueeze(-1)
        imag = x_fft.imag * real_w.unsqueeze(-1) + x_fft.real * imag_w.unsqueeze(-1)

        x_filtered = torch.fft.irfft(torch.complex(real, imag), n=L, dim=2)
        return x_filtered * torch.sigmoid(x)

# ======================= Experts =======================
class DepthWiseExpert(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size//2, groups=in_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class FrequencyAwareMoE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.expert_small = DepthWiseExpert(in_dim, out_dim, 3)
        self.expert_medium = DepthWiseExpert(in_dim, out_dim, 7)
        self.expert_large = DepthWiseExpert(in_dim, out_dim, 15)

        self.expert_spectral = SpectralGatingUnit(in_dim)
        self.spectral_proj = nn.Conv1d(in_dim, out_dim, 1)

        # global gate (more stable than token-wise)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_dim, 4, 1),
            nn.Softmax(dim=1)
        )

        self.residual_proj = nn.Conv1d(in_dim, out_dim, 1)
        self.post = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        gate = self.gate(x)  # (B,4,1)

        out_small = self.expert_small(x) * gate[:, 0:1]
        out_medium = self.expert_medium(x) * gate[:, 1:2]
        out_large = self.expert_large(x) * gate[:, 2:3]

        spectral = self.spectral_proj(self.expert_spectral(x)) * gate[:, 3:4]

        fused = out_small + out_medium + out_large + spectral
        x = fused + self.residual_proj(x)
        x = self.post(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return x

        

class CrossModalAttentionGate(nn.Module):
    """
    Gated cross-modal attention for drug-target interaction learning.

    Uses cross-attention with confidence gating: the model learns when to trust
    cross-modal information vs unimodal features for each drug-target pair.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cross_d2t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_t2d = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.conf_d = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid()
        )
        self.conf_t = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid()
        )
        self.norm_d = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)

    def forward(self, d_feat, t_feat):
        d = d_feat.unsqueeze(1)
        t = t_feat.unsqueeze(1)

        d_cross, _ = self.cross_d2t(d, t, t)
        t_cross, _ = self.cross_t2d(t, d, d)
        d_cross, t_cross = d_cross.squeeze(1), t_cross.squeeze(1)

        cd = self.conf_d(torch.cat([d_feat, d_cross], dim=-1))
        ct = self.conf_t(torch.cat([t_feat, t_cross], dim=-1))

        d_out = self.norm_d(d_feat + cd * d_cross)
        t_out = self.norm_t(t_feat + ct * t_cross)
        return d_out, t_out


class SubspaceRoutingFusion(nn.Module):
    """
    Subspace Routing Fusion (SRF) — 原创融合模块

    将特征空间划分为多个竞争子空间，每个子空间拥有独立的变换参数。
    路由器为每个特征维度生成软分配权重，不同于 MoE 的 token 级路由，
    SRF 在维度级别进行路由，使不同子空间专注于不同的特征交互模式。
    配合子空间重要性权重，实现输入自适应的分治融合策略。
    """
    def __init__(self, dim, num_subspaces=6):
        super().__init__()
        self.num_subspaces = num_subspaces

        # 每个子空间的独立变换
        self.subspace_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim//4),
                nn.GELU(),
                nn.Linear(dim//4, dim)
            )
            for _ in range(num_subspaces)
        ])

        # 维度级路由器：为每个特征维度分配子空间权重
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim * num_subspaces)
        )

        # 子空间重要性评估
        self.importance = nn.Sequential(
            nn.Linear(dim, num_subspaces),
            nn.Softmax(dim=-1)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        B, D = x.shape

        # 维度级路由权重
        route_logits = self.router(x).view(B, D, self.num_subspaces)
        route_weights = F.softmax(route_logits, dim=-1)  # (B, D, K)

        # 子空间重要性
        imp = self.importance(x)  # (B, K)

        # 子空间变换 + 维度级加权组合
        output = 0
        for k in range(self.num_subspaces):
            transformed = self.subspace_transforms[k](x)       # (B, D)
            routed = transformed * route_weights[:, :, k]      # 维度级门控
            output = output + routed * imp[:, k:k+1]           # 子空间级加权

        return self.norm(identity + output)


# ======================= Full Model =======================
class FSR_DTA(nn.Module):
    def __init__(self, n_drug, n_target, hidden_dim=128, dropout=0.):
        super().__init__()

        self.drug_embed = nn.Embedding(n_drug, hidden_dim)
        self.target_embed = nn.Embedding(n_target, hidden_dim)

        self.drug_cnn = FrequencyAwareMoE(hidden_dim, hidden_dim)
        self.target_cnn = FrequencyAwareMoE(hidden_dim, hidden_dim)

        self.cross_attn = CrossModalAttentionGate(hidden_dim)

        self.descriptor = nn.Sequential(
            nn.Linear(147+20+400+8000, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim)
        )

        self.fingerprint = nn.Sequential(
            nn.Linear(2048+315+881+166+36, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim)
        )

        self.fusion = SubspaceRoutingFusion(hidden_dim*4)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        drug_seq, target_seq, desc, fp = data

        d_seq = self.drug_embed(drug_seq).transpose(1, 2)
        t_seq = self.target_embed(target_seq).transpose(1, 2)

        d_feat = self.drug_cnn(d_seq)
        t_feat = self.target_cnn(t_seq)
        
        de = self.descriptor(desc)
        fp = self.fingerprint(fp)
        d_feat, t_feat = self.cross_attn(d_feat, t_feat)


        feat = torch.cat([d_feat, t_feat, de, fp], dim=-1)
        feat = self.fusion(feat)
        return self.head(feat)