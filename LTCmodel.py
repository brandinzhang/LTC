import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ================= Utilities =================
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]


# ================= SupConLoss =================
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, device="cuda"):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

# ================= CrossAttentionMapper =================
class CrossAttentionMapper(nn.Module):
    """Generate pseudo text embeddings from visual embeddings via cross-attention."""
    def __init__(self, clip_model, d_img=512, d_txt=512, n_kv=8, n_queries=16, n_heads=8, ff_ratio=4):
        super().__init__()
        self.clip_text = clip_model.transformer.eval()
        for block in self.clip_text.resblocks:
            block.attn_mask = None
        for p in self.clip_text.parameters():
            p.requires_grad = False

        self.proj_k = nn.Linear(d_img, d_txt * n_kv)
        self.proj_v = nn.Linear(d_img, d_txt * n_kv)
        self.q_tokens = nn.Parameter(torch.randn(n_queries, d_txt) / (d_txt ** 0.5))
        self.attn = nn.MultiheadAttention(d_txt, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_txt)
        self.ff = nn.Sequential(
            nn.Linear(d_txt, ff_ratio * d_txt),
            nn.GELU(),
            nn.Linear(ff_ratio * d_txt, d_txt),
        )
        self.ln2 = nn.LayerNorm(d_txt)
        self.pos_embed = nn.Parameter(torch.randn(n_queries, d_txt) / (d_txt ** 0.5))
        self.n_kv = n_kv
        self.d_txt = d_txt

    def forward(self, img_feat):
        B = img_feat.size(0)
        K = self.proj_k(img_feat).view(B, self.n_kv, self.d_txt)
        V = self.proj_v(img_feat).view(B, self.n_kv, self.d_txt)

        Q = self.q_tokens.unsqueeze(0).expand(B, -1, -1)
        Q = Q + self.pos_embed.unsqueeze(0)

        Z, _ = self.attn(self.ln1(Q), self.ln1(K), self.ln1(V))
        Z = Z + Q
        Z2 = self.ff(self.ln2(Z))
        tokens = Z + Z2  # [B, n_queries, d_txt]

        tokens = tokens.permute(1, 0, 2)  # [seq, batch, dim]
        clip_dtype = next(self.clip_text.parameters()).dtype
        tokens = tokens.to(clip_dtype)
        out = self.clip_text(tokens)
        sent = out[0]
        return F.normalize(sent.float(), dim=-1)

# ================= Multimodal fusion head =================
class FusionHead(nn.Module):
    def __init__(self, v_dim, t_dim, out_dim, hidden=1024, fusion_type='concat', num_heads=4):
        super().__init__()
        assert fusion_type in ['mean', 'concat', 'attn']
        self.fusion_type = fusion_type
        self.num_heads = num_heads

        if fusion_type == 'concat':
            in_dim = v_dim + t_dim
            self.fuse = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim)
            )
        elif fusion_type == 'attn':
            in_dim = max(v_dim, t_dim)
            self.q_proj = nn.Linear(v_dim, in_dim)
            self.k_proj = nn.Linear(t_dim, in_dim)
            self.v_proj = nn.Linear(t_dim, in_dim)
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim)
            )

    def forward(self, f_v, f_t):
        if self.fusion_type == 'mean':
            z = (f_v + f_t) / 2
        elif self.fusion_type == 'concat':
            z = self.fuse(torch.cat([f_v, f_t], dim=-1))
        else:  # 'attn'
            q = self.q_proj(f_v).unsqueeze(1)
            k = self.k_proj(f_t).unsqueeze(1)
            v = self.v_proj(f_t).unsqueeze(1)
            attn = torch.softmax((q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5), dim=-1)
            z = (attn @ v).squeeze(1)
            z = self.fc(z)
        return l2_normalize(z, dim=-1)

# ================= Prototype layer =================
class PrototypeLayer(nn.Module):
    def __init__(self, num_classes, feat_dim=512, include_unknown=False, init_scale=20.0):
        super().__init__()
        total = num_classes + (1 if include_unknown else 0)
        proto_u = torch.randn(total, feat_dim)
        self.proto_u = nn.Parameter(proto_u)
        self.include_unknown = include_unknown
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, feats: torch.Tensor):
        P = F.normalize(self.proto_u, dim=-1)   # [K, D]
        Fm = F.normalize(feats, dim=-1)         # [B, D]
        logits = self.scale * (Fm @ P.T)
        return logits

    @torch.no_grad()
    def add_prototype(self, new_proto: torch.Tensor, normalize: bool = True, device=None):
        if normalize:
            new_proto = F.normalize(new_proto, dim=-1)
        if new_proto.ndim == 1:
            new_proto = new_proto.unsqueeze(0)
        if device is not None:
            new_proto = new_proto.to(device)
        new_Pu = torch.cat([self.proto_u.data, new_proto], dim=0)
        self.proto_u = nn.Parameter(new_Pu)

    @torch.no_grad()
    def remove_unknown(self):
        if self.include_unknown:
            self.proto_u = nn.Parameter(self.proto_u[:-1])
            self.include_unknown = False

# ================= LTCModel =================
class LTCModel(nn.Module):
    """
    Switches:
      When args.use_visual_only = True, skip the text mapper and fusion head, and use v as the classification feature.
      The prototype head uses feat_dim = args.v_dim. The returned dict keeps 'm' = v_emb for backward compatibility.
    """
    def __init__(self, args, clip_model):
        super().__init__()
        self.clip = clip_model
        self.use_visual_only = True

        # Freeze most visual parameters; only train the last block, etc. (per the original setup)
        for p in self.clip.visual.parameters():
            p.requires_grad = False
        if hasattr(self.clip.visual, "transformer") and len(self.clip.visual.transformer.resblocks) > 0:
            for p in self.clip.visual.transformer.resblocks[-1].parameters():
                p.requires_grad = True
        if hasattr(self.clip.visual, "ln_post"):
            for p in self.clip.visual.ln_post.parameters():
                p.requires_grad = True
        if hasattr(self.clip.visual, "proj") and self.clip.visual.proj is not None:
            self.clip.visual.proj.requires_grad_(True)

        # Build mapper & fusion only when needed
        if not self.use_visual_only:
            self.text_mapper = CrossAttentionMapper(
                clip_model, d_img=args.v_dim, d_txt=args.t_dim,
                n_kv=getattr(args, "n_kv", 8),
                n_queries=getattr(args, "n_queries", 16),
                n_heads=getattr(args, "n_heads", 8),
                ff_ratio=getattr(args, "ff_ratio", 4),
            )
            self.fusion = FusionHead(args.v_dim, args.t_dim, args.m_dim,
                                     hidden=args.fusion_hidden, fusion_type=args.fusion_type)
            feat_dim = args.m_dim
        else:
            # Visual-only: no mapper/fusion needed
            self.text_mapper = None
            self.fusion = None
            feat_dim = args.v_dim

        self.proto = PrototypeLayer(args.known_num_classes, feat_dim=feat_dim, include_unknown=False)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        v = self.clip.encode_image(images).float()
        return F.normalize(v, dim=-1)

    def forward(self, images: torch.Tensor, t_override: Optional[torch.Tensor] = None):
        v_emb = self.encode_image(images)  # [B, Dv]

        if self.use_visual_only:
            f_m = v_emb                      # directly use visual features
            logits = self.proto(f_m)
            return {"v": v_emb, "t_hat": None, "t_used": None, "m": f_m, "logits": logits}

        # Otherwise, use the multimodal branch (for non-baseline experiments)
        t_hat = self.text_mapper(v_emb)                # [B, Dt]
        if t_override is not None:
            t_used = F.normalize(t_override, dim=-1)
        else:
            t_used = t_hat
        v_emb = l2_normalize(v_emb, dim=-1)
        t_used = l2_normalize(t_used, dim=-1)
        f_m = self.fusion(v_emb, t_used)
        logits = self.proto(f_m)
        return {"v": v_emb, "t_hat": t_hat, "t_used": t_used, "m": f_m, "logits": logits}
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1]*n
    def find(self, x):
        while self.p[x]!=x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a,b):
        pa,pb = self.find(a), self.find(b)
        if pa==pb: return False
        if self.sz[pa] < self.sz[pb]:
            pa,pb = pb,pa
        self.p[pb] = pa
        self.sz[pa] += self.sz[pb]
        return True

@torch.no_grad()
def merge_dense_components_blockwise(P, counts, start_idx, s_th, tuner=None, block=2048):

    device = P.device
    C = P.size(0)
    M = C - start_idx
    if M <= 1:
        return P, counts


    nn_idx = torch.full((M,), -1, dtype=torch.long, device=device)
    nn_sim = torch.full((M,), -2.0, dtype=torch.float32, device=device)

    Q = P[start_idx:]  # [M,D]
    for a0 in range(0, M, block):
        a1 = min(M, a0 + block)
        QA = Q[a0:a1]                     # [Ba,D]
        best_sim = torch.full((a1-a0,), -2.0, device=device)
        best_j   = torch.full((a1-a0,), -1,   dtype=torch.long, device=device)
        for b0 in range(0, M, block):
            b1 = min(M, b0 + block)
            QB = Q[b0:b1]                 # [Bb,D]
            S  = QA @ QB.t()              # [Ba,Bb]
            if a0 == b0:
                diag = torch.arange(0, a1-a0, device=device)
                S[diag, diag] = -2.0      # exclude self
            cur_sim, cur_j = torch.max(S, dim=1)   # [Ba]
            mask = cur_sim > best_sim
            best_sim[mask] = cur_sim[mask]
            best_j[mask]   = cur_j[mask] + b0
        nn_sim[a0:a1] = best_sim
        nn_idx[a0:a1] = best_j

    edges = []
    for i in range(M):
        j = int(nn_idx[i].item())
        if j < 0: continue
        # mutual nearest neighbors
        if int(nn_idx[j].item()) == i:
            s = float(nn_sim[i].item())
            if tuner is not None:
                tuner.observe_merge_pair(s)
            if s >= float(s_th):
                if i < j:
                    edges.append((i, j))

    if len(edges) == 0:
        return P, counts

    # DSU merging
    dsu = DSU(M)
    for i, j in edges:
        dsu.union(i, j)

    # Build connected components and compute count-weighted means
    comp = {}  # root -> list of nodes
    for i in range(M):
        r = dsu.find(i)
        comp.setdefault(r, []).append(i)

    keep_mask = torch.ones(C, dtype=torch.bool, device=device)
    for nodes in comp.values():
        if len(nodes) <= 1:
            continue
        glob_idx = [start_idx + n for n in nodes]
        ws = counts[glob_idx].unsqueeze(1)                   # [k,1]
        center = torch.sum(P[glob_idx] * ws, dim=0) / torch.clamp(ws.sum(), 1e-6)
        center = torch.nn.functional.normalize(center, dim=-1)
        P[glob_idx[0]] = center
        counts[glob_idx[0]] = counts[glob_idx].sum()
        # mark the other nodes for deletion
        for g in glob_idx[1:]:
            keep_mask[g] = False

    P = P[keep_mask]
    counts = counts[keep_mask]
    return P, counts
