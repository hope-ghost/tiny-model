"""Decoder-only GPT。现代小模型架构：RoPE 位置编码 + RMSNorm + SwiGLU + GQA。
代码尽量精简并加中文注释，方便学习和修改。"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """RMSNorm：只按均方根缩放，不减均值、无 bias。比 LayerNorm 更省算、更稳。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 在最后一维（通道）上算均方根，用 float32 保证数值稳定
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """预计算 RoPE 的 cos/sin 表。形状 (seq_len, head_dim)。"""
    # 每两维一组，频率随维度指数衰减（低维转得快、高维转得慢）
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)              # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (T, head_dim)，前后半重复
    return emb.cos(), emb.sin()


def rotate_half(x):
    """把最后一维前后两半旋转 90°：[x1, x2] -> [-x2, x1]。RoPE 的核心操作。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """给 q、k 施加旋转位置编码。q,k: (B, n_head, T, head_dim)；cos,sin: (T, head_dim)。"""
    cos = cos[None, None, :, :].to(q.dtype)   # 广播到 batch 和 head 维
    sin = sin[None, None, :, :].to(q.dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class CausalSelfAttention(nn.Module):
    """因果自注意力 + GQA：多个 Q 头共享少数几个 KV 头，省显存、推理更快。"""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head 必须能被 n_kv_head 整除"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "RoPE 需要 head_dim 为偶数"
        # Q 投影到全部头；K、V 只投影到较少的 KV 头（GQA 的关键）
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.size()  # batch, 序列长, 通道
        # 拆成多头：(B, n_head, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        # 施加旋转位置编码（对 q、k 生效）
        q, k = apply_rope(q, k, cos, sin)
        # GQA：把少数 KV 头复制到与 Q 头数一致，再走标准注意力
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # PyTorch 内置 flash-attention，is_causal 自动加因果掩码
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 合并多头
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """SwiGLU 前馈网络：gate 分支用 SiLU 激活后与 up 分支逐元素相乘，再降回原维。
    比传统 GELU-MLP 表达力更强，是现代 LLM 的标配。"""

    def __init__(self, cfg):
        super().__init__()
        # 隐藏维取 8/3 * n_embd（对齐 SwiGLU 参数量惯例），再向上取整到 64 的倍数
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)  # 门控分支
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)    # 升维分支
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)  # 降回原维
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """一个 Transformer 层：注意力 + 前馈，都用 pre-norm(RMSNorm) 残差连接。"""

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)  # 残差：注意力
        x = x + self.mlp(self.ln_2(x))             # 残差：前馈
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token 嵌入（位置信息由 RoPE 提供）
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=RMSNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 权重绑定：输入嵌入和输出投影共享参数，省显存又提效果
        self.transformer.wte.weight = self.lm_head.weight
        # 预计算 RoPE 表并注册为 buffer（persistent=False：不进 checkpoint，随模型迁移设备）
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(cfg.block_size, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # 残差投影层特殊初始化（GPT-2 做法，稳定深层训练）
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # 权重绑定后 wte 与 lm_head 是同一份参数，sum 不会重复计入
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"序列长 {T} 超过 block_size {self.cfg.block_size}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]   # 取前 T 个位置的旋转表
        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)
        if targets is not None:
            # 训练：算所有位置的 loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        # 推理：只需最后一个位置的预测
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        # 2 维以上参数（矩阵）做权重衰减，1 维参数（RMSNorm 权重等）不做
        decay, no_decay = [], []
        for p in self.parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        use_fused = device_type == 'cuda'
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=use_fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        """自回归续写：给定开头 idx，逐个 token 生成。"""
        for _ in range(max_new_tokens):
            # 上下文超长就只保留最后 block_size 个 token
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 温度越高越随机
            if top_k is not None:
                # 只在概率最高的 top_k 个候选里采样，避免生成低质 token
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
