import logging
import math
from typing import Dict, List, Optional
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from utils import SmoothedValue
from utils import split_cluster_acc_v1, split_cluster_acc_v2
from LTCmodel import SupConLoss
from mkee import MKEEGenerator  # added: MKEE
from LTCmodel import merge_dense_components_blockwise
# ================= Utility functions =================
def _build_label_mapping_if_needed(targets: torch.Tensor, known_classes: List[int]) -> torch.Tensor:
    K = len(known_classes)
    if targets.max().item() < K and targets.min().item() >= 0:
        return targets
    mapping = {g: i for i, g in enumerate(known_classes)}
    mapped = torch.tensor([mapping[int(t.item())] for t in targets], device=targets.device, dtype=targets.dtype)
    return mapped


@torch.no_grad()
def build_ncm_prototypes(train_loader_singleview, model, device, known_classes, save_path: str = None):
    """
    Build visual class means (NCM) using the single-view training set, consistent with evaluation geometry.
    """
    model.eval()
    K = len(known_classes)
    D = int(model.proto.proto_u.size(1))
    sums = torch.zeros(K, D, device=device)
    counts = torch.zeros(K, dtype=torch.long, device=device)

    for images, label, *rest in tqdm(train_loader_singleview, desc="Build NCM"):
        images = images.to(device, non_blocking=True)
        label  = label.to(device, non_blocking=True)
        mapped = _build_label_mapping_if_needed(label, known_classes)
        out = model(images)
        v   = F.normalize(out["v"].float(), dim=-1)  # train/eval operate on pure cosine similarity
        for k in range(K):
            mask = (mapped == k)
            if mask.any():
                sums[k]   += v[mask].sum(dim=0)
                counts[k] += int(mask.sum().item())

    C = sums
    nz = counts > 0
    if nz.any():
        C[nz] = F.normalize(C[nz] / counts[nz].unsqueeze(1), dim=-1)
    if (~nz).any():
        fallback = F.normalize(model.proto.proto_u.data.to(device)[(~nz).nonzero(as_tuple=True)[0]], dim=-1)
        C[(~nz)] = fallback
    if save_path:
        torch.save({"means": C.detach().cpu(), "counts": counts.cpu()}, save_path)
    return C  # [K, D]


@torch.no_grad()
def evaluate(test_loader, model, device, known_classes,
             tau: float = None,              # None: no growing; otherwise threshold-based growing
             print_logits: bool = False,
             protos_override: torch.Tensor = None,
             topK=None
             ) -> Dict[str, float]:
    """
    Evaluate with nearest-prototype classification using NCM/overridden prototypes. If tau is not None,
    samples with max similarity < tau will trigger prototype growing.
    This version additionally performs DSU merging only on the newly-grown prototype subset to suppress
    fragmentation and uncontrolled class/prototype explosion.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    K = len(known_classes)

    # ---------------- DSU merge config (only applied to newly-grown prototypes) ----------------
    ENABLE_MERGE = True          # set False to disable merging
    S_MERGE      = 0.995         # mutual-NN similarity threshold (higher is safer)
    MERGE_EVERY  = 1000          # try merging every N seen samples
    MERGE_BLOCK  = 2048          # block size for similarity scan to avoid O(M^2) VRAM blow-up

    # Use externally provided NCM (recommended)
    if protos_override is not None:
        P = F.normalize(protos_override.to(device), dim=-1)   # [C,D]
    else:
        P_u = model.proto.proto_u.data
        P = F.normalize(P_u, dim=-1).to(device)               # [C,D]

    # Counts are used for weighted averaging during merges (initialize known prototypes with larger weight)
    counts = torch.ones(P.size(0), device=device, dtype=torch.float32) * 10.0

    preds, targets, mask_old = [], [], []
    logged_topk = [] if print_logits else None
    init_proto_count = int(P.size(0))     # boundary: only merge prototypes grown after this point
    new_class_count = 0

    seen = 0
    for images, label, *rest in tqdm(test_loader, desc="Eval(Baseline+Merge/NCM)"):
        images = images.to(device, non_blocking=True)
        label  = label.to(device, non_blocking=True)

        out = model(images)
        V = F.normalize(out["m"].float(), dim=-1)  # in visual-only mode, m == v
        B = V.size(0)

        targets.extend(label.tolist())
        mask_old.extend((label < K).tolist())

        for i in range(B):
            f_i = V[i:i+1]             # [1, D]
            sims = (f_i @ P.T)         # [1, C_curr]
            max_sim, pred = sims.max(dim=-1)
            pred_i = int(pred.item())

            if print_logits:
                vec = sims.squeeze(0)                 # [C_curr]
                k = min(5, vec.numel())
                topv, topi = torch.topk(vec, k=k, dim=-1)
                logged_topk.append((topi.detach().cpu().numpy(), topv.detach().cpu().numpy()))

            # Threshold-based growing (cosine similarity)
            if (tau is not None) and (float(max_sim.item()) < tau):
                new_vec = f_i[0].detach()            # [D]
                P = torch.cat([P, new_vec.unsqueeze(0)], dim=0)
                counts = torch.cat([counts, torch.tensor([1.0], device=device)], dim=0)
                new_class_count += 1
                pred_i = P.size(0) - 1

            preds.append(pred_i)
            seen += 1

            # Periodically run DSU merging on the newly-grown prototype subset (does not affect known prototypes)
            if ENABLE_MERGE and (seen % MERGE_EVERY == 0) and (P.size(0) > init_proto_count + 1):
                P, counts = merge_dense_components_blockwise(
                    P, counts,
                    start_idx=init_proto_count,   # merge newly-grown prototypes only
                    s_th=S_MERGE,
                    tuner=None,                   # no auto-tuning/observer
                    block=MERGE_BLOCK
                )

    preds1  = np.array(preds, dtype=np.int64)
    targets = np.array([int(x) for x in targets], dtype=np.int64)
    mask    = np.array(mask_old,       dtype=bool)

    all1, old1, new1 = split_cluster_acc_v1(y_true=targets, y_pred=preds1, mask=mask, topK=topK)
    all2, old2, new2 = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask, topK=topK)

    pretty = (
        f"[V1] all={all1:.3f} old={old1:.3f} new={new1:.3f}\n"
        f"[V2] all={all2:.3f} old={old2:.3f} new={new2:.3f}\n"
        f"[Grow/Merge] new_classes={new_class_count} "
        f"(protos: {init_proto_count}→{P.size(0)})"
    )
    print(pretty)

    if print_logits and len(logged_topk) > 0:
        np.set_printoptions(suppress=True, precision=4, linewidth=120)
        print("[Logits] Per-sample top-5 (index, value):")
        for idx, (inds, vals) in enumerate(logged_topk):
            pairs = " ".join([f"({int(inds[j])},{vals[j]:.4f})" for j in range(len(vals))])
            print(f"  sample#{idx:05d}: {pairs}")

    return {
        "all_acc_v1": float(all1), "old_acc_v1": float(old1), "new_acc_v1": float(new1),
        "all_acc_v2": float(all2), "old_acc_v2": float(old2), "new_acc_v2": float(new2),
        "new_classes": int(new_class_count),
        "pretty": pretty,
    }



def train_one_epoch(model,
                    data_loader,
                    optimizer,
                    device,
                    epoch,
                    max_norm,
                    args=None,
                    set_training_mode=True) -> Dict[str, float]:
    """
    Training objective:
      L = CE + alpha * SupCon + gamma_mm * ( L_pos(τ) + L_neg(τ) )
    where:
      - τ is an online-calibrated threshold via "quantile target + EMA" (no backprop through τ);
      - L_pos(τ) = ReLU((τ + m_pos) - s_max(pos)), encouraging known samples to have s_max >= τ + m_pos;
      - L_neg(τ) = ReLU(s_max(ood) - (τ - m_neg)), encouraging pseudo-unknown samples to have s_max <= τ - m_neg;
      - s_max = max_k <v, C_k>, where C_k is the running class mean (unit-normalized), consistent with eval geometry;
      - MKEE: disabled for warmup epochs, then inject a batch of x_ood with probability p_mkee.
    """
    logger = logging.getLogger("train")
    model.train(set_training_mode)

    # ===== meters =====
    loss_ce_meter   = SmoothedValue(window_size=20, fmt='{value:.4f}')
    loss_supcon_meter = SmoothedValue(window_size=20, fmt='{value:.4f}')
    loss_pos_meter  = SmoothedValue(window_size=20, fmt='{value:.4f}')
    loss_neg_meter  = SmoothedValue(window_size=20, fmt='{value:.4f}')
    lr_meter        = SmoothedValue(window_size=1,  fmt='{value:.6f}')

    # ===== hyper (with defaults to avoid touching the parser) =====
    alpha      = float(getattr(args, "alpha", 0.5))
    gamma_mm   = float(getattr(args, "gamma_mm", 0.05))
    q_pos      = float(getattr(args, "q_pos", 0.20))   # conservative: lower quantile for known
    q_neg      = float(getattr(args, "q_neg", 0.80))   # conservative: higher quantile for unknown
    m_pos      = float(getattr(args, "m_pos", 0.05))
    m_neg      = float(getattr(args, "m_neg", 0.05))
    beta_ema   = float(getattr(args, "tau_beta", 0.01))
    p_mkee     = float(getattr(args, "p_mkee", 0.20))
    warm_mkee  = 1
    # Initialize τ if this is the first training run
    if not hasattr(args, "tau_ema"):
        args.tau_ema = float(getattr(args, "tau_train", 0.65))

    # ===== Per-epoch running class-mean statistics (consistent with above) =====
    K = len(args.known_classes)
    D = int(model.proto.proto_u.size(1))
    sums = torch.zeros(K, D, device=device)
    counts = torch.zeros(K, dtype=torch.float32, device=device)
    fallback = F.normalize(model.proto.proto_u.data.to(device), dim=-1)

    def current_centroids():
        C = fallback.clone()
        nz = counts > 0
        if nz.any():
            C[nz] = F.normalize(sums[nz] / counts[nz].unsqueeze(1), dim=-1)
        return C

    # ===== SupCon =====
    supcon = SupConLoss(temperature=getattr(args, "supcon_t", 0.07))

    # ===== MKEE (optional) =====
    mkee = MKEEGenerator(
        model        = model,
        alpha        = float(getattr(args, "mkee_alpha", 0.4)),
        eps          = float(getattr(args, "mkee_eps",   0.05)),
        sigma        = float(getattr(args, "mkee_sigma", 0.5)),
        lam1         = float(getattr(args, "mkee_lam1",  0.1)),
        sigma_mult   = float(getattr(args, "mkee_sigma_mult", 1.0)),
        l2norm_feats = True,
        device       = str(device),
    )

    logger.info("Start training epoch %d | tau_ema_init=%.4f", epoch, args.tau_ema)
    pbar = tqdm(data_loader, desc=f"Epoch [{epoch}]", ncols=140, leave=True)

    for batch in pbar:
        # ------- Load batch -------
        if isinstance(batch, (list, tuple)):
            samples, targets = batch[0], batch[1]
        else:
            samples, targets = batch

        if isinstance(samples, list):
            x1 = samples[0].to(device, non_blocking=True)
            x2 = samples[1].to(device, non_blocking=True)
        else:
            x1 = x2 = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mapped = _build_label_mapping_if_needed(targets, args.known_classes)

        # ------- Forward -------
        out1 = model(x1)  # {'v','m','logits',...}; in visual-only mode, m == v
        out2 = model(x2)

        # CE (supervision via the prototype head)
        loss_ce = F.cross_entropy(out1["logits"], mapped)

        # v1 = F.normalize(out1["v"], dim=-1)
        # loss_cos = arcface_ce_from_protos(
        #     v_norm=v1,
        #     proto_u=model.proto.proto_u,
        #     target=mapped,
        #     scale=model.proto.scale,                         # reuse the head's scale
        #     margin=float(getattr(args, "arcface_margin", 0.30))
        # )

        # SupCon (two views, visual features)
        feats = torch.stack([out1["v"], out2["v"]], dim=1)
        loss_supcon = supcon(feats, labels=mapped, device=str(device))

        # ------- Compute s_max(pos) using current class centroids -------
        with torch.no_grad():
            C = current_centroids()              # [K, D]
        v_pos = F.normalize(out1["v"], dim=-1)   # [B, D]
        S_pos = v_pos @ C.t()                    # [B, K]
        smax_pos = S_pos.max(dim=1).values       # [B], s_max for known samples

        # ------- MKEE: per-batch probability + warmup to generate an OOD batch -------
        has_neg = (epoch >= warm_mkee) and (float(torch.rand(1)) < p_mkee)
        if has_neg:
            with torch.no_grad():                 # reference feature distribution (within the batch)
                v_ref = v_pos.detach()
            x_ood = mkee.generate(x1.detach(), feats_ref=v_ref)       # [B,3,H,W]
            out_ood = model(x_ood)
            v_ood = F.normalize(out_ood["v"], dim=-1)
            S_neg = v_ood @ C.t()
            smax_neg = S_neg.max(dim=1).values   # [B]
        else:
            smax_neg = None

        # ------- Hinge: push known above the threshold, pull unknown below (using current tau_ema) -------
        tau = float(args.tau_ema)
        loss_pos = F.relu((tau + m_pos) - smax_pos).mean()
        if smax_neg is not None:
            loss_neg = F.relu(smax_neg - (tau - m_neg)).mean()
        else:
            loss_neg = torch.tensor(0.0, device=device)

        loss_mm = loss_pos + loss_neg

        # ------- Total loss -------
        loss = loss_ce + alpha * loss_supcon + gamma_mm * loss_mm

        # ------- Backprop & update -------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if (max_norm is not None) and (max_norm > 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # ------- Accumulate class-mean stats for this batch (update after use to avoid leakage) -------
        with torch.no_grad():
            sums.index_add_(0, mapped, v_pos)
            counts.index_add_(0, mapped, torch.ones_like(mapped, dtype=torch.float32))

        # ------- Quantile + EMA update for tau (only when neg samples exist) -------
        if smax_neg is not None:
            u_pos = torch.quantile(smax_pos.detach(), q_pos).item()
            u_neg = torch.quantile(smax_neg.detach(), q_neg).item()
            tau_tgt = 0.5 * ((u_pos - m_pos) + (u_neg + m_neg))
            # Optional robustness: clamp to [-1, 1]
            tau_tgt = max(-1.0, min(1.0, tau_tgt))
            args.tau_ema = (1.0 - beta_ema) * args.tau_ema + beta_ema * tau_tgt

        # ------- Logging -------
        loss_ce_meter.update(float(loss_ce))
        loss_supcon_meter.update(float(loss_supcon))
        loss_pos_meter.update(float(loss_pos))
        loss_neg_meter.update(float(loss_neg))
        lr_meter.update(optimizer.param_groups[0]["lr"])

        pbar.set_postfix({
            "ce": f"{loss_ce_meter.global_avg:.4f}",
            "supcon": f"{loss_supcon_meter.global_avg:.4f}",
            "mm_pos": f"{loss_pos_meter.global_avg:.4f}",
            "mm_neg": f"{loss_neg_meter.global_avg:.4f}",
            "tau": f"{args.tau_ema:.3f}",
            "lr": f"{lr_meter.global_avg:.6f}",
        })

    print(f"Averaged stats - ce: {loss_ce_meter.global_avg:.4f} | "
          f"supcon: {loss_supcon_meter.global_avg:.4f} | "
          f"mm_pos: {loss_pos_meter.global_avg:.4f} | "
          f"mm_neg: {loss_neg_meter.global_avg:.4f} | "
          f"tau_ema: {args.tau_ema:.4f} | "
          f"lr: {lr_meter.global_avg:.6f}")

    return {
        "loss_ce": loss_ce_meter.global_avg,
        "loss_supcon": loss_supcon_meter.global_avg,
        "loss_mm_pos": loss_pos_meter.global_avg,
        "loss_mm_neg": loss_neg_meter.global_avg,
        "tau_ema": float(args.tau_ema),
        "lr": lr_meter.global_avg
    }
