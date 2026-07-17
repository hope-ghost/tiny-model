"""Decoder-only GPT。结构与 GPT-2 一致，代码尽量精简并加中文注释。"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力：每个位置只能看到自己和前面的 token。"""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        # 一次性算出 q,k,v 三份投影，效率更高
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.size()  # batch, 序列长, 通道
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # 拆成多头：(B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # PyTorch 内置 flash-attention，is_causal 自动加因果掩码
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 合并多头
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """前馈网络：先升维 4 倍，GELU 激活，再降回来。"""

    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """一个 Transformer 层：注意力 + 前馈，都用 pre-norm 残差连接。"""

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差：注意力
        x = x + self.mlp(self.ln_2(x))   # 残差：前馈
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token 嵌入
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # 位置嵌入
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 权重绑定：输入嵌入和输出投影共享参数，省显存又提效果
        self.transformer.wte.weight = self.lm_head.weight
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
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel()  # 习惯上不计位置嵌入

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"序列长 {T} 超过 block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
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
        # 2 维以上参数（矩阵）做权重衰减，1 维参数（bias/LayerNorm）不做
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
