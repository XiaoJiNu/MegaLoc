"""MegaLoc: One Retrieval to Place Them All

This module implements the MegaLoc model for visual place recognition.
The model combines a Vision Transformer backbone with an optimal transport-based
feature aggregation module.

Paper: https://arxiv.org/abs/2502.17237
License: MIT

中文讲解(含实测维度/参数量): doc/reproduction/20260915_153138_explained_v2.md
一句话总览: 输入 RGB 图 [B,3,H,W] → DINOv2 ViT-B/14 骨干提特征
→ SALAD 式最优传输(OT)聚合头压成 [B,16640] → Linear 降到 [B,8448] → L2 归一化;
两张图的相似度 = 描述子点积(余弦)。总参数约 228.6M(骨干 86.6M + 聚合头 142.1M)。
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tfm


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    # Sinkhorn 迭代(log 域): 交替做行/列 logsumexp 归一, 把打分矩阵 M 变成
    # 满足给定边际分布的运输计划。u/v 是两组对偶变量。
    M = M / reg  # regularization  # reg 是熵正则温度

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    # 返回 log 域的运输计划 [B, m+1, n](含 dustbin 行)
    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    # S: 原始打分 [B, m, n], 本项目中 m=64 簇, n=patch 数(322 输入时 n=529)。
    # 作用: 把原始分数双随机化成合法运输计划 —— 每个 patch(列)分给
    # 64 簇+dustbin 的比例合计为 1, 每个簇(行)接收总量也被均衡,
    # 防止"所有 patch 涌进同一个簇"的塌缩(比逐 patch softmax 强)。
    batch_size, m, n = S.size()
    # augment scores matrix
    # 补一行 dustbin(垃圾桶) → [B, m+1, n]: 无信息 patch(天空/模糊块)进桶, 不进任何簇
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    # 目标边际取均匀分布 -log(n+m); dustbin 行额外加 log(n-m) 修正 → 桶容量更大
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    # MegaLoc/SALAD 用 3 轮 Sinkhorn 就够, 不需迭代到收敛
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm  # 减 norm 还原量纲, 返回 log 域计划 [B, m+1, n]


class FeatureAggregator(nn.Module):
    """Optimal transport-based aggregation of local features into global descriptor.

    This module aggregates local patch features into a compact global representation
    using differentiable optimal transport.

    中文: OT 软聚类聚合头。骨干给出 n 个局部 patch 向量 + 1 个全局 cls 向量,
    本模块不是简单平均(会稀释小目标), 而是"带垃圾桶的软聚类记账":
    预定义 num_clusters 个语义簇(簇由训练学出), 用 Sinkhorn 最优传输决定
    每个 patch 往哪些簇各记多少账, 64 个簇的加权平均向量再与 cls 路拼接。
    对视角变化、局部遮挡更鲁棒。

    Args:
        num_channels: Number of input feature channels (from backbone)
        num_clusters: Number of cluster centers
        cluster_dim: Dimensionality of cluster descriptors
        token_dim: Dimensionality of global scene token
        mlp_dim: Hidden dimension for MLPs
        dropout: Dropout probability (0 to disable)
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,  # 注意: 这是 SALAD 原论文配置; MegaLoc 经 agg_config 显式传 256
        token_dim=256,
        mlp_dim=512,
        dropout=0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.mlp_dim = mlp_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token
        # 全局路: cls token [B,768] → [B,token_dim], 整图摘要向量
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.token_dim)
        )
        # MLP for local features
        # 内容路: 每个 patch "长什么样"(记账内容)。
        # 用 1×1 Conv 作用在特征图上 = 逐 patch 独立 Linear, 省一次 reshape。
        # [B,768,23,23] → flatten → [B,cluster_dim,529]
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.cluster_dim, 1),
        )
        # MLP for score matrix
        # 打分路: 每个 patch 对 64 簇的原始亲和分 [B,768,23,23] → [B,num_clusters,529]
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.num_clusters, 1),
        )
        # Dustbin parameter
        # 可学习标量: dustbin 行(垃圾桶)对所有 patch 的固定打分
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Args:
            x: Tuple of (features, token)
                features: [B, C, H, W] spatial feature map
                token: [B, C] global CLS token

        Returns:
            Global descriptor [B, num_clusters * cluster_dim + token_dim]
        """
        # 以 MegaLoc 默认配置 + 322×322 输入为例:
        # x=[B,768,23,23] 特征图(529 个 patch), t=[B,768] cls token
        x, t = x

        f = self.cluster_features(x).flatten(2)  # 内容 [B,256,529]
        p = self.score(x).flatten(2)  # 打分 [B,64,529]
        t = self.token_features(t)  # 全局摘要 [B,256]

        # Sinkhorn OT → log 域计划 [B,65,529], exp 还原为概率,
        # 再删掉 dustbin 行(第 65 行) —— 进桶的"垃圾"不进任何簇的账
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]  # [B,64,529]

        # einsum: 簇 k 向量 = Σ_n f[:,n]·p[k,n], 即成员 patch 内容的加权和
        # [B,256,529] × [B,64,529] → [B,64,256]
        # 分段归一(cls 一次、每簇 256 维各自一次、整体再一次)
        # 防止某簇权重范数过大垄断相似度
        f = torch.cat(
            [
                F.normalize(t, p=2, dim=-1),  # [B,256]
                F.normalize(torch.einsum("bdn,bkn->bdk", f, p), p=2, dim=1).flatten(1),  # [B,16384]
            ],
            dim=-1,
        )  # [B,16640]

        return F.normalize(f, p=2, dim=-1)


# ==============================================================================
# Vision Transformer Components
# ==============================================================================


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings using a convolutional layer.

    中文: 切块。kernel=stride=14 的 Conv 把每个 14×14 像素块"读"成一个
    embed_dim 维向量(等价 Linear(14·14·3=588→768))。
    322×322 输入 → 23×23=529 个 token; 518(训练分辨率) → 37×37=1369 个。
    """

    def __init__(self, image_size: int = 518, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B,3,H,W] → [B,768,h,w], h=H/14
        x = x.flatten(2)  # → [B,768,n], 行优先展平(与 DINOv2.forward 的还原对称)
        x = x.transpose(1, 2)  # → [B,n,768] token 序列
        return x


class LayerScale(nn.Module):
    """Learnable per-channel scaling as used in CaiT and DINOv2.

    中文: 可学习的逐通道乘法 x*γ(初值 1e-5), 让训练初期每层接近恒等映射。
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module.

    中文: 多头自注意力。dim=768, 12 头 × head_dim 64。
    一个 Linear(768→2304) 同时出 Q/K/V, 走 F.scaled_dot_product_attention
    (内部 1/√64 缩放, 数学上即 softmax(QKᵀ/8)V)。
    """

    def __init__(
        self, dim: int, num_heads: int = 12, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # 322 输入时 N=530(529 patch + 1 cls)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)  # 拼回 [B,N,768]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """MLP module with GELU activation.

    中文: ViT 前馈层, mlp_ratio=4 时 768 → 3072(GELU) → 768。
    """

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Vision Transformer block with LayerScale.

    中文: Pre-Norm ViT 层(CaiT/DINOv2 风格), 残差结构:
        x = x + LayerScale(Attn(LayerNorm(x)))
        x = x + LayerScale(MLP(LayerNorm(x)))
    维度全程不变: [B,530,768] 进出, 主干共 12 层。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_value=init_values)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, init_value=init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2(nn.Module):
    """DINOv2 Vision Transformer backbone for feature extraction.

    This implements a ViT-B/14 architecture compatible with DINOv2 weights.

    中文: 特征提取骨干, 标准 ViT-B/14(权重兼容官方 DINOv2),
    只是为"输出空间特征图 + cls token 两路"做了适配。
    输入 [B,3,H,W] → 输出 (patch特征图 [B,768,H/14,W/14], cls [B,768])。
    """

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_channels = embed_dim  # 聚合头按此取输入通道数(768)

        self.patch_embed = PatchEmbedding(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )

        # 位置编码插值参数: offset=0.1 是平滑偏移项, 见 interpolate_pos_encoding
        self.interpolate_offset = 0.1
        self.interpolate_antialias = False

        # cls token: 全局摘要占位符, 靠自注意力把整图信息聚合到自己位置
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置嵌入按 518 分辨率训练: 37×37=1369 patch + 1 cls = 1370 个
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolate positional encoding for different input sizes.

        中文: 换分辨率不重训的关键。位置编码存的是 37×37 格(518 分辨率),
        输入 322×322 时只有 23×23 格 → 把 patch 位置嵌入 reshape 成
        [1,768,37,37] 的"图像", 双三次插值到 [1,768,23,23] 再摊平,
        与 cls 的位置编码拼接。所以骨干对任意(14 倍数)分辨率是弹性的。
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1  # 当前输入的 patch 数(不含 cls)
        N = self.pos_embed.shape[1] - 1  # 训练时的 patch 数(1369)

        if npatch == N and w == h:
            return self.pos_embed  # 分辨率一致, 直接用

        pos_embed = self.pos_embed.float()  # 插值统一在 fp32 做
        class_pos_embed = pos_embed[:, 0]  # cls 的位置编码, 不参与插值
        patch_pos_embed = pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = w // self.patch_size  # 目标网格宽(格数)
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # 训练网格边长(37)

        # +0.1 偏移: 插值采样点略微外扩, 平滑边缘格的位置编码
        sx = float(w0 + self.interpolate_offset) / M
        sy = float(h0 + self.interpolate_offset) / M

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert (w0, h0) == patch_pos_embed.shape[-2:]  # 插值后格数必须精确匹配
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from images.

        Args:
            images: Input images [B, 3, H, W] where H, W are multiples of 14

        Returns:
            Tuple of (patch_features [B, 768, H//14, W//14], cls_token [B, 768])
        """
        B, _, H, W = images.shape

        x = self.patch_embed(images)  # [B, n, 768], 322 输入时 n=529
        cls_tokens = self.cls_token.expand(B, -1, -1)  # batch 内共享, expand 不复制内存
        x = torch.cat((cls_tokens, x), dim=1)  # cls 拼在序列头 → [B, 530, 768]
        x = x + self.interpolate_pos_encoding(x, H, W)  # 加(必要时插值过的)位置编码

        for block in self.blocks:  # 12 层 Transformer, 形状不变
            x = block(x)

        x = self.norm(x)  # 最后一层 LayerNorm

        cls_token = x[:, 0]  # [B,768] 全局摘要 → 聚合头全局路
        patch_tokens = x[:, 1:]  # [B,529,768]
        # 按行优先还原 token→网格(与 patch_embed 的 flatten(2) 对称),
        # 变回空间特征图 [B,768,23,23] 供聚合头的 1×1 Conv 使用
        patch_features = patch_tokens.reshape(B, H // self.patch_size, W // self.patch_size, self.embed_dim).permute(
            0, 3, 1, 2
        )

        return patch_features, cls_token


# ==============================================================================
# Main Model
# ==============================================================================


class L2Norm(nn.Module):
    """中文: L2 归一化层。归一后描述子点积 = 余弦相似度。"""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


class Aggregator(nn.Module):
    """中文: 聚合头包装层 = FeatureAggregator(OT 聚合 → [B,16640]) + Linear 降维。

    MegaLoc 相对 SALAD 论文的唯一结构改动: 加这层 Linear 把 16640 压到 8448(=66×128)。
    fp32 下一条描述子 33KB, 比 SALAD 原生省一半存储/检索带宽;
    8448 就是建库检索时的向量维度。降维矩阵 16640×8448≈140.6M, 是全模型参数大头。
    """

    def __init__(self, feat_dim, agg_config, salad_out_dim):
        super().__init__()
        self.agg = FeatureAggregator(**agg_config)  # OT 聚合 → [B, salad_out_dim=16640]
        self.linear = nn.Linear(salad_out_dim, feat_dim)  # 降维瓶颈 16640 → 8448

    def forward(self, x):
        x = self.agg(x)
        return self.linear(x)


class MegaLoc(nn.Module):
    """MegaLoc: Unified visual place recognition model.

    Combines a DINOv2 Vision Transformer backbone with optimal transport-based
    feature aggregation to produce compact, discriminative image descriptors
    for place recognition and image retrieval tasks.

    中文: 视觉位置识别(VPR)主模型。结构:
        图像 [B,3,H,W] ─DINOv2→ (patch特征图 [B,768,h,w], cls [B,768])
                     ─FeatureAggregator(OT软聚类)→ [B,16640]
                     ─Linear→ [B,8448] ─L2→ 描述子(‖d‖₂=1)
    检索用法: sim = query @ gallery.T(点积即余弦), top-K 命中地图节点,
    其建图位姿即定位结果。

    Args:
        feat_dim: Output descriptor dimensionality (default: 8448)
        num_clusters: Number of cluster centers for aggregation (default: 64)
        cluster_dim: Dimensionality of cluster descriptors (default: 256)
        token_dim: Dimensionality of global scene token (default: 256)
        mlp_dim: Hidden dimension for MLPs (default: 512)

    Example:
        >>> model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        >>> model.eval()
        >>> descriptor = model(image)  # [B, 8448]
    """

    def __init__(
        self,
        feat_dim: int = 8448,
        num_clusters: int = 64,
        cluster_dim: int = 256,
        token_dim: int = 256,
        mlp_dim: int = 512,
    ):
        super().__init__()

        self.backbone = DINOv2()  # ViT-B/14 骨干, 86.6M 参数
        # salad_out_dim = 64×256 + 256 = 16640: 聚合头拼接后的维度
        self.salad_out_dim = num_clusters * cluster_dim + token_dim
        self.aggregator = Aggregator(
            feat_dim=feat_dim,
            agg_config={  # 显式覆盖 FeatureAggregator 的类默认值(如 cluster_dim 128→256)
                "num_channels": self.backbone.num_channels,
                "num_clusters": num_clusters,
                "cluster_dim": cluster_dim,
                "token_dim": token_dim,
                "mlp_dim": mlp_dim,
            },
            salad_out_dim=self.salad_out_dim,
        )
        self.feat_dim = feat_dim
        self.l2norm = L2Norm()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global descriptor from images.

        Args:
            images: Input images [B, 3, H, W]
                    (调用方需先做 ToTensor + ImageNet mean/std 归一化)

        Returns:
            L2-normalized descriptors [B, feat_dim]
        """
        b, c, h, w = images.shape
        # 兜底: 分辨率必须被 patch_size=14 整除(Conv 切块不留边角),
        # 否则自动 resize 到最近的 14 倍数 —— 任意尺寸都能出 [B,8448]
        if h % 14 != 0 or w % 14 != 0:
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            images = tfm.resize(images, [h, w], antialias=True)
        features = self.aggregator(self.backbone(images))  # 骨干 → 聚合头 → [B,8448]
        # L2Norm: 输入 [B,8448] → 输出 [B,8448](形状不变)。
        # 计算: 每行(一条描述子)除以自己的 L2 范数, 即 x[i] / sqrt(Σ x[i,j]²)(dim=1),
        # 归一后每行范数=1, 两描述子点积 = 余弦相似度
        features = self.l2norm(features)  # 最终 features: [B, feat_dim] = [B, 8448]
        return features
