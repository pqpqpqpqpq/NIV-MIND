"""Fine-grained ST-GCN--AltFormer backbone for NIV-MIND."""

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DropPath(nn.Module):
    """Apply stochastic depth per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or (self.drop_prob is None) or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def create_ventilator_adjacency_matrix(num_nodes=8):
    """Build the normalized adjacency matrix for ventilator channels."""
    # Channel order: FiO2, PEEPset, PI, I:E, VT, MV, RR, Ppeak.
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes): adj[i, i] = 1.0

    adj[0, 1] = 0.3; adj[0, 7] = 0.3  # FiO2
    adj[1, 0] = 0.3; adj[1, 2] = 0.8; adj[1, 7] = 0.9 # PEEPset
    adj[2, 1] = 0.8; adj[2, 4] = 0.7; adj[2, 7] = 0.9 # PI
    adj[3, 4] = 0.7; adj[3, 6] = 0.6  # I:E
    adj[4, 2] = 0.7; adj[4, 3] = 0.7; adj[4, 5] = 1.0; adj[4, 6] = 0.9; adj[4, 7] = 0.8 # VT
    adj[5, 4] = 1.0; adj[5, 6] = 1.0 # MV
    adj[6, 3] = 0.6; adj[6, 4] = 0.9; adj[6, 5] = 1.0 # RR
    adj[7, 0] = 0.3; adj[7, 1] = 0.9; adj[7, 2] = 0.9; adj[7, 4] = 0.8 # Ppeak
    
    adj = adj + np.eye(num_nodes)
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return torch.FloatTensor(adj_normalized)

class LearnableAdjacency(nn.Module):
    """Learnable adjacency matrix with optional residual parameterization."""
    def __init__(self, num_nodes, mode='residual', init_adj=None, sparse_ratio=0.3, symmetric=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.mode = mode
        self.sparse_ratio = sparse_ratio
        self.symmetric = symmetric
        
        if init_adj is not None:
            self.register_buffer('fixed_adj', init_adj)
        else:
            self.register_buffer('fixed_adj', torch.eye(num_nodes))
            
        if mode == 'residual':
            self.adj_residual = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        elif mode == 'fully_learnable':
            self.adj_params = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
            
    def normalize_adj(self, adj):
        d = adj.sum(dim=1).clamp(min=1e-12)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def get_adjacency(self):
        if self.mode == 'residual':
            adj = self.fixed_adj + torch.tanh(self.residual_weight) * self.adj_residual
        elif self.mode == 'fully_learnable':
            adj = self.adj_params
        else:
            adj = self.fixed_adj

        if self.symmetric:
            adj = (adj + adj.t()) / 2
        adj = adj + torch.eye(self.num_nodes, device=adj.device)
        
        if self.sparse_ratio > 0 and self.training:
            k = int(self.num_nodes * self.num_nodes * (1 - self.sparse_ratio))
            _, indices = torch.topk(adj.abs().reshape(-1), k)
            mask = torch.zeros_like(adj.reshape(-1))
            mask[indices] = 1
            adj = (adj.reshape(-1) * mask).view(self.num_nodes, self.num_nodes)
            
        return self.normalize_adj(adj)

    def get_regularization_loss(self):
        if self.mode == 'residual':
            return torch.abs(self.adj_residual).mean() + \
                   (torch.norm(self.adj_residual - self.adj_residual.t(), p='fro') if self.symmetric else 0)
        return torch.tensor(0., device=self.fixed_adj.device)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias: self.bias = nn.Parameter(torch.Tensor(out_features))
        else: self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None: nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None: output = output + self.bias
        return output

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel_size=9, stride=1, dropout=0.1):
        super().__init__()
        pad = (temporal_kernel_size - 1) // 2
        self.tcn1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (temporal_kernel_size, 1), (stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.gcn = GraphConvolution(out_channels, out_channels)
        self.bn_gcn = nn.BatchNorm2d(out_channels)
        self.tcn2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (temporal_kernel_size, 1), (stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        self.residual = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        res = self.residual(x)
        x = self.tcn1(x)
        
        B, C, T, N = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B * T, N, C)
        x = self.gcn(x, adj)
        x = x.view(B, T, N, C).permute(0, 3, 1, 2).contiguous()
        x = self.bn_gcn(x)
        x = self.relu(x)
        
        x = self.tcn2(x)
        x = self.relu(x + res)
        return x

class TrainableSTGCN_FeaturePreserving(nn.Module):
    """Map ``[B, L, F]`` inputs to ``[B, L, F, D]`` features."""
    def __init__(self, num_features=8, feature_dim=64, seq_len=960, num_blocks=3, adj_mode='residual'):
        super().__init__()
        self.num_features = num_features
        
        init_adj = create_ventilator_adjacency_matrix(num_features)
        self.learnable_adj = LearnableAdjacency(num_features, mode=adj_mode, init_adj=init_adj)
        
        mid_dim = max(8, feature_dim // 4)
        self.input_projection = nn.Sequential(nn.Linear(1, mid_dim), nn.ReLU())
        self.input_bn = nn.BatchNorm1d(mid_dim)
        
        self.stgcn_blocks = nn.ModuleList([
            STGCNBlock(
                in_channels=mid_dim if i == 0 else feature_dim,
                out_channels=feature_dim
            ) for i in range(num_blocks)
        ])
        
        self.feature_enhance = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
            for _ in range(num_features)
        ])
        self.temporal_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        B, L, F = x.shape
        adj = self.learnable_adj.get_adjacency()

        x = x.unsqueeze(-1).permute(0, 2, 1, 3).reshape(B * F, L, 1)
        x = self.input_projection(x)
        x = x.transpose(1, 2); x = self.input_bn(x); x = x.transpose(1, 2)
        x = x.reshape(B, F, L, -1).permute(0, 3, 2, 1)

        for block in self.stgcn_blocks:
            x = block(x, adj)

        x = x.permute(0, 2, 3, 1)

        enhanced = []
        for i in range(F):
            feat = x[:, :, i, :]
            feat = self.feature_enhance[i](feat)
            attn, _ = self.temporal_attention(feat, feat, feat)
            enhanced.append(self.output_norm(feat + 0.5 * attn))
        
        return torch.stack(enhanced, dim=2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x_attn, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, None

class Enhanced_STAltFormer_3D_NoMonitor(nn.Module):
    def __init__(self, num_frame=16, num_features=8, feature_dim=64,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=2.,
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0.2, output_dim=512):
        super().__init__()
        self.num_frame = num_frame
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.feature_projection = nn.Linear(feature_dim, embed_dim)

        self.st_spatial_pos_embed = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        self.st_spatial_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate * i / depth, norm_layer=norm_layer)
            for i in range(depth // 2)])
        self.st_temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.st_temporal_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate * i / depth, norm_layer=norm_layer)
            for i in range(depth // 2)])

        self.ts_temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.ts_temporal_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate * i / depth, norm_layer=norm_layer)
            for i in range(depth // 2)])
        self.ts_spatial_pos_embed = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        self.ts_spatial_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate * i / depth, norm_layer=norm_layer)
            for i in range(depth // 2)])

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.st_norm = norm_layer(embed_dim)
        self.ts_norm = norm_layer(embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, output_dim),
            norm_layer(output_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            norm_layer(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

        nn.init.trunc_normal_(self.st_spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.st_temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.ts_temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.ts_spatial_pos_embed, std=0.02)

    def st_forward(self, x):
        B, T, F, E = x.shape
        xs = rearrange(x, 'b t f e -> (b t) f e')
        xs = self.pos_drop(xs + self.st_spatial_pos_embed)
        for blk in self.st_spatial_blocks: xs, _ = blk(xs)
        xs = rearrange(xs, '(b t) f e -> b t f e', b=B)
        xt = xs.mean(dim=2) + self.st_temporal_pos_embed
        xt = self.pos_drop(xt)
        for blk in self.st_temporal_blocks: xt, _ = blk(xt)
        return self.st_norm(xt)

    def ts_forward(self, x):
        B, T, F, E = x.shape
        xt = rearrange(x, 'b t f e -> (b f) t e')
        xt = self.pos_drop(xt + self.ts_temporal_pos_embed)
        for blk in self.ts_temporal_blocks: xt, _ = blk(xt)
        xt = rearrange(xt, '(b f) t e -> b t f e', b=B)
        xs = xt.mean(dim=1) + self.ts_spatial_pos_embed
        xs = self.pos_drop(xs)
        for blk in self.ts_spatial_blocks: xs, _ = blk(xs)
        x_ts = self.ts_norm(xs.mean(dim=1))
        return x_ts.unsqueeze(1).expand(-1, T, -1)

    def forward(self, x):
        x = self.feature_projection(x)
        x_st = self.st_forward(x)
        x_ts = self.ts_forward(x)
        w = F.softmax(self.fusion_weights, dim=0)
        x_cat = torch.cat([x_st * w[0], x_ts * w[1]], dim=-1)
        x_fused = self.fusion(x_cat)
        return self.output_projection(x_fused.mean(dim=1))
