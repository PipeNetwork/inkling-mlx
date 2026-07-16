"""Parity for the vision (HMLP fold+project) and audio (dMel embed+sum) towers."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlx.core as mx

from inkling_mlx.config import VisionConfig, AudioConfig
from inkling_mlx.vision import VisionModel, plan_out_scales
from inkling_mlx.audio import AudioModel


class RefRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        dt = x.dtype; x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def fold(x, t_fold, hw_fold):
    B, T, H, W, C = x.shape
    tn, hn, wn = T // t_fold, H // hw_fold, W // hw_fold
    x = x.reshape(B, tn, t_fold, hn, hw_fold, wn, hw_fold, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(B, tn, hn, wn, t_fold * hw_fold * hw_fold * C)


class RefVision(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        scales = plan_out_scales(cfg.temporal_patch_size, cfg.patch_size, cfg.n_layers, cfg.num_channels)
        self.folds = []
        self.lin = nn.ModuleList(); self.norms = nn.ModuleList()
        for i in range(cfg.n_layers):
            s, e = scales[i], scales[i + 1]
            shuffle = (e[0]//s[0])*(e[1]//s[1])*(e[2]//s[2])
            in_dim = int(s[3])*int(shuffle)
            add_norm = i != cfg.n_layers - 1
            out_dim = cfg.text_hidden_size if not add_norm else int(e[3])
            self.lin.append(nn.Linear(in_dim, out_dim, bias=False))
            self.norms.append(RefRMSNorm(out_dim) if add_norm else nn.Identity())
            self.folds.append((int(e[0]//s[0]), int(e[1]//s[1]), add_norm))
        self.final_norm = RefRMSNorm(cfg.text_hidden_size)

    def forward(self, x):
        n = x.shape[0]
        for i, (tf, hf, add_norm) in enumerate(self.folds):
            if hf > 1 or tf > 1:
                x = fold(x, tf, hf)
            x = self.lin[i](x)
            if add_norm:
                x = self.norms[i](x); x = F.gelu(x)
        x = self.final_norm(x)
        return x.reshape(n, -1)


def t2m(t): return mx.array(t.detach().float().numpy())


def main():
    torch.manual_seed(0); np.random.seed(0)
    # ---- vision ----
    vc = VisionConfig(text_hidden_size=6144, patch_size=40, temporal_patch_size=2, num_channels=3, n_layers=4)
    ref = RefVision(vc).eval()
    for p in ref.parameters():
        if p.dim() >= 2: torch.nn.init.normal_(p, std=0.05)
    mlxv = VisionModel(vc)
    upd = {}
    for i in range(vc.n_layers):
        upd[f"layers.linear_{i}.weight"] = t2m(ref.lin[i].weight)
        if not isinstance(ref.norms[i], nn.Identity):
            upd[f"layers.norm_{i}.weight"] = t2m(ref.norms[i].weight)
    upd["final_norm.weight"] = t2m(ref.final_norm.weight)
    from mlx.utils import tree_unflatten
    mlxv.update(tree_unflatten(list(upd.items()))); mx.eval(mlxv.parameters())

    pix = np.random.randn(2, 2, 40, 40, 3).astype(np.float32)
    with torch.no_grad():
        rv = ref(torch.tensor(pix)).float().numpy()
    mv = np.array(mlxv(mx.array(pix)).astype(mx.float32))
    dv = np.abs(rv - mv)
    print(f"[vision] out {rv.shape}  max|Δ|={dv.max():.3e} mean|Δ|={dv.mean():.3e}")

    # ---- audio ----
    ac = AudioConfig(text_hidden_size=6144, n_mel_bins=80, mel_vocab_size=16)
    emb = nn.Embedding(ac.n_mel_bins * ac.mel_vocab_size, ac.text_hidden_size)
    torch.nn.init.normal_(emb.weight, std=0.05)
    anorm = RefRMSNorm(ac.text_hidden_size)
    offsets = torch.arange(ac.n_mel_bins) * ac.mel_vocab_size

    mlxa = AudioModel(ac)
    from mlx.utils import tree_unflatten
    mlxa.update(tree_unflatten([("encoder.weight", t2m(emb.weight)), ("final_norm.weight", t2m(anorm.weight))]))
    mx.eval(mlxa.parameters())

    frames = np.random.randint(0, 16, size=(5, 80))
    with torch.no_grad():
        ra = anorm(emb(torch.tensor(frames) + offsets).sum(dim=-2)).float().numpy()
    ma = np.array(mlxa(mx.array(frames)).astype(mx.float32))
    da = np.abs(ra - ma)
    print(f"[audio ] out {ra.shape}  max|Δ|={da.max():.3e} mean|Δ|={da.mean():.3e}")

    ok = dv.max() < 1e-2 and da.max() < 1e-2
    print("\nMM PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
