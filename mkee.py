# mkee.py
import torch
import torch.nn.functional as F

class MKEEGenerator:
    """
    Entropy–Kernel Guided Pseudo-Sample Generator (MKEE)
    A generic implementation that can be plugged into any vision model to generate pseudo-unknown samples.
    Supports automatic sigma adjustment, L2 feature normalization, and gradient normalization.
    """

    def __init__(self, model,
                 alpha=0.4, eps=0.05, sigma=0.5, lam1=0.1,
                 sigma_mult=1.0, l2norm_feats=True, device="cuda"):
        """
        Args:
            model: model that outputs (logits, feats)
            alpha: Beta distribution parameter (controls mixup)
            eps: perturbation step size
            sigma: initial RBF kernel bandwidth
            lam1: kernel density loss weight
            sigma_mult: sigma scaling factor (e.g., 1.0 means median distance)
            l2norm_feats: whether to L2-normalize features
        """
        self.model = model
        self.alpha = alpha
        self.eps = eps
        self.sigma = sigma
        self.lam1 = lam1
        self.sigma_mult = sigma_mult
        self.l2norm_feats = l2norm_feats
        self.device = device

    @staticmethod
    def rbf_rho(feats_x, feats_ref, sigma=0.5):
        dist = torch.cdist(feats_x, feats_ref)
        return torch.exp(-dist**2 / (2 * sigma**2)).mean()

    def sample_mixup(self, x):
        idx = torch.randperm(x.size(0), device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(x.device)
        return lam * x + (1 - lam) * x[idx]

    def generate(self, x_batch, feats_ref=None):
        """
        Given a batch of raw samples x_batch, return pseudo-unknown samples x_ood.
        """
        x_mix = self.sample_mixup(x_batch).detach().clone().requires_grad_(True)
        logits_mix = self.model(x_mix)
        logits = logits_mix['logits']
        feats_x = logits_mix['v']

        if self.l2norm_feats:
            feats_x = F.normalize(feats_x, dim=1)
            if feats_ref is not None:
                feats_ref = F.normalize(feats_ref, dim=1)

        # dynamic sigma
        sigma_use = self.sigma
        if self.sigma_mult is not None and feats_ref is not None:
            with torch.no_grad():
                d_med = torch.cdist(feats_x, feats_ref).median().clamp_min(1e-12)
            sigma_use = float(self.sigma_mult) * d_med.item()

        # entropy
        p = torch.softmax(logits, dim=1)
        H = -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean()

        # kernel density
        if feats_ref is None:
            feats_ref = feats_x.detach()
        rho = self.rbf_rho(feats_x, feats_ref.detach(), sigma=sigma_use)
        
        # gradient ascent direction
        obj = H - self.lam1 * rho
        g = torch.autograd.grad(obj, x_mix, retain_graph=False)[0]
        g_norm = g.norm(p=2, dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        x_ood = (x_mix + self.eps * (g / g_norm)).detach()

        return x_ood

"""
from mkee import MKEEGenerator

mkee = MKEEGenerator(model,
                     alpha=0.4, eps=0.05, sigma=0.5,
                     lam1=0.1, sigma_mult=1.0, l2norm_feats=True)

for imgs, _ in dataloader:
    imgs = imgs.to(device)
    with torch.no_grad():
        _, feats_ref = model(imgs, return_feat=True)
    x_ood = mkee.generate(imgs, feats_ref)
"""
