"""Standalone PyTorch reference for the Inkling TEXT model, extracted from
transformers PR #47347 (modular_inkling.py) with framework deps stubbed out
(no Cache, no masking_utils, no kernel hub). Prefill only. Used purely to
numerically validate the MLX port on a tiny random config.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = float("-inf")


def repeat_kv(x, n):
    b, kv, s, d = x.shape
    if n == 1:
        return x
    x = x[:, :, None, :, :].expand(b, kv, n, s, d)
    return x.reshape(b, kv * n, s, d)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


class RelativeLogits(nn.Module):
    def __init__(self, d_rel, rel_extent):
        super().__init__()
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent))

    def forward(self, relative_states, q_pos, k_pos):
        rel_logits = (relative_states @ self.proj).transpose(1, 2)
        distance = (q_pos[:, None] - k_pos[None, :])[None, None, :, :]
        gather = distance.clamp(0, self.rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
        bias = rel_logits.gather(-1, gather)
        return bias.masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)


class ShortConv(nn.Module):
    def __init__(self, channels, k):
        super().__init__()
        self.k = k
        self.conv1d = nn.Conv1d(channels, channels, k, groups=channels, padding=k - 1, bias=False)

    def forward(self, x):  # x: [B, L, C]
        dt = x.dtype
        x = x.float()
        residual = x
        seq = x.shape[1]
        x = x.transpose(1, 2)
        x = F.conv1d(x, self.conv1d.weight.float(), None, padding=self.k - 1, groups=x.shape[1])[:, :, :seq]
        x = x.transpose(1, 2)
        return (x + residual).to(dt)


class Attention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.is_sliding = cfg["layer_types"][layer_idx] == "hybrid_sliding"
        self.head_dim = cfg["swa_head_dim"] if self.is_sliding else cfg["head_dim"]
        self.num_heads = cfg["swa_num_attention_heads"] if self.is_sliding else cfg["num_attention_heads"]
        self.num_kv = cfg["swa_num_key_value_heads"] if self.is_sliding else cfg["num_key_value_heads"]
        self.n_rep = self.num_heads // self.num_kv
        self.sliding_window = cfg["sliding_window_size"] if self.is_sliding else None
        self.rel_extent = cfg["sliding_window_size"] if self.is_sliding else cfg["rel_extent"]
        self.d_rel = cfg["d_rel"]
        self.scaling = 1.0 / self.head_dim
        h = cfg["hidden_size"]
        self.wq_du = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.wk_dv = nn.Linear(h, self.num_kv * self.head_dim, bias=False)
        self.wv_dv = nn.Linear(h, self.num_kv * self.head_dim, bias=False)
        self.wr_du = nn.Linear(h, self.num_heads * self.d_rel, bias=False)
        self.wo_ud = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.k_sconv = ShortConv(self.num_kv * self.head_dim, cfg["sconv_kernel_size"])
        self.v_sconv = ShortConv(self.num_kv * self.head_dim, cfg["sconv_kernel_size"])
        self.q_norm = RMSNorm(self.head_dim, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, cfg["rms_norm_eps"])
        self.rel_logits_proj = RelativeLogits(self.d_rel, self.rel_extent)

    def forward(self, x):
        B, L, _ = x.shape
        hs = (B, L, -1, self.head_dim)
        q = self.wq_du(x)
        k = self.k_sconv(self.wk_dv(x))
        v = self.v_sconv(self.wv_dv(x))
        rel = self.wr_du(x)
        q = self.q_norm(q.view(hs)).transpose(1, 2)
        k = self.k_norm(k.view(hs)).transpose(1, 2)
        v = v.view(hs).transpose(1, 2)
        q_pos = torch.arange(L)
        k_pos = torch.arange(L)
        rel = rel.view(B, L, self.num_heads, -1)
        position_bias = self.rel_logits_proj(rel, q_pos, k_pos)
        if not self.is_sliding and self.cfg["log_scaling_n_floor"] is not None:
            eff = (q_pos + 1).float()
            tau = 1.0 + self.cfg["log_scaling_alpha"] * torch.log(
                (eff / self.cfg["log_scaling_n_floor"]).clamp(min=1.0))
            tau = tau.view(1, 1, -1, 1)
            q = (q.float() * tau).to(q.dtype)
            position_bias = (position_bias.float() * tau).to(position_bias.dtype)
        # additive mask
        dist = q_pos[:, None] - k_pos[None, :]
        allowed = dist >= 0
        if self.sliding_window is not None:
            allowed = allowed & (dist < self.sliding_window)
        amask = torch.where(allowed, 0.0, torch.tensor(NEG_INF))[None, None]
        kk = repeat_kv(k, self.n_rep)
        vv = repeat_kv(v, self.n_rep)
        aw = (q.float() @ kk.transpose(2, 3).float()) * self.scaling
        aw = aw + position_bias.float() + amask.float()
        aw = F.softmax(aw, dim=-1)
        out = aw.to(vv.dtype) @ vv
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo_ud(out)


class DenseMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, inter = cfg["hidden_size"], cfg["dense_intermediate_size"]
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)
        self.global_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) * self.global_scale


class Router(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ne = cfg["n_routed_experts"]
        self.ns = cfg["n_shared_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        self.route_scale = cfg["route_scale"]
        self.weight = nn.Parameter(torch.empty(self.ne + self.ns, cfg["hidden_size"]))
        self.global_scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(self.ne))  # e_score_correction_bias

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        logits = F.linear(flat, self.weight)
        scores = logits.sigmoid()
        routed_scores = scores[..., : self.ne]
        choice = routed_scores + self.bias
        idx = torch.topk(choice, self.top_k, dim=-1, sorted=False)[1]
        routed_logits = logits[..., : self.ne]
        shared_logits = logits[..., self.ne:]
        topk_logits = torch.cat([routed_logits.gather(-1, idx), shared_logits], dim=-1)
        lp = F.logsigmoid(topk_logits)
        w = torch.exp(lp - torch.logsumexp(lp, dim=-1, keepdim=True))
        w = w * self.route_scale * self.global_scale
        shared_g = w[..., -self.ns:].contiguous()
        w = w[..., : self.top_k].contiguous()
        return w, idx, shared_g


class Experts(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ne = cfg["n_routed_experts"]
        inter = cfg["moe_intermediate_size"]
        self.gate_up_proj = nn.Parameter(torch.empty(self.ne, 2 * inter, cfg["hidden_size"]))
        self.down_proj = nn.Parameter(torch.empty(self.ne, cfg["hidden_size"], inter))

    def forward(self, x, idx, w):
        out = torch.zeros_like(x)
        mask = F.one_hot(idx, self.ne).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = e[0]
            pos, tok = torch.where(mask[e])
            cur = x[tok]
            gate, up = F.linear(cur, self.gate_up_proj[e]).chunk(2, dim=-1)
            ch = F.silu(gate) * up
            ch = F.linear(ch, self.down_proj[e]) * w[tok, pos, None]
            out.index_add_(0, tok, ch.to(out.dtype))
        return out


class SharedExperts(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ns = cfg["n_shared_experts"]
        inter = cfg["moe_intermediate_size"]
        self.gate_proj = nn.Parameter(torch.empty(self.ns, inter, cfg["hidden_size"]))
        self.up_proj = nn.Parameter(torch.empty(self.ns, inter, cfg["hidden_size"]))
        self.down_proj = nn.Parameter(torch.empty(self.ns, cfg["hidden_size"], inter))

    def forward(self, x, gammas):
        shape = x.shape
        x = x.reshape(1, -1, shape[-1]).expand(self.ns, -1, -1)
        gammas = gammas.reshape(-1, self.ns, 1).transpose(0, 1)
        gate = torch.bmm(x, self.gate_proj.transpose(1, 2))
        up = torch.bmm(x, self.up_proj.transpose(1, 2))
        act = F.silu(gate) * up * gammas
        down = torch.bmm(act, self.down_proj.transpose(1, 2))
        return down.float().sum(0).to(x.dtype).view(shape)


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = Router(cfg)
        self.experts = Experts(cfg)
        self.shared_experts = SharedExperts(cfg)

    def forward(self, x):
        residual = x
        shape = x.shape
        w, idx, shared_g = self.gate(x)
        x = x.view(-1, x.shape[-1])
        x = self.experts(x, idx, w).view(*shape)
        x = x + self.shared_experts(residual, shared_g)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.attn = Attention(cfg, i)
        self.attn_norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp_norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = MoE(cfg) if cfg["mlp_layer_types"][i] == "sparse" else DenseMLP(cfg)
        self.attn_sconv = ShortConv(cfg["hidden_size"], cfg["sconv_kernel_size"])
        self.mlp_sconv = ShortConv(cfg["hidden_size"], cfg["sconv_kernel_size"])

    def forward(self, x):
        r = x
        h = self.attn_norm(x)
        h = self.attn(h)
        h = self.attn_sconv(h)
        x = r + h
        r = x
        h = self.mlp_norm(x)
        h = self.mlp(h)
        h = self.mlp_sconv(h)
        x = r + h
        return x


class TextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.embed_norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.unembed = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def forward(self, ids):
        h = self.embed_norm(self.embed(ids))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        h = h / self.cfg["logits_mup_width_multiplier"]
        logits = self.unembed(h)
        uv = self.cfg.get("unpadded_vocab_size")
        if uv is not None and uv < logits.shape[-1]:
            logits = logits[..., :uv]
        return logits
