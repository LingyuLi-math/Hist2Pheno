########################################################
# 2026.10.24, add adjacent hierarchy losses / 相邻层级监督与一致性
# Keep three independent CE heads; every extra term can be disabled.
########################################################
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HierarchyConfig:
    l2_l12_nll: float = 0.5
    l12_l1_nll: float = 0.5
    l2_l12_js: float = 0.1
    l12_l1_js: float = 0.1

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError('Hierarchy weights must be finite and nonnegative')


def parent_map(children, parents, n_children):
    """Only use explicit ontology/train labels; never infer parents from val labels."""
    out = np.full(n_children, -1, dtype=np.int64)
    for c, p in zip(children, parents):
        if out[int(c)] not in (-1, int(p)):
            raise ValueError('Adjacent hierarchy is not a single-parent tree')
        out[int(c)] = int(p)
    if (out < 0).any():
        raise ValueError('Incomplete hierarchy; supply full ontology for missing classes')
    return out


def aggregation_matrix(mapping, n_parents, device):
    mapping = torch.as_tensor(mapping, dtype=torch.long, device=device)
    if ((mapping < 0) | (mapping >= n_parents)).any():
        raise ValueError('Invalid parent index')
    return F.one_hot(mapping, n_parents).float()


def _js(p, q):
    p, q = p.clamp_min(1e-7), q.clamp_min(1e-7)
    m = (p + q) / 2
    return ((p * (p.log()-m.log())).sum(1) + (q * (q.log()-m.log())).sum(1)).mean()/2


def adjacent_loss(logits2, logits12, logits1, y12, y1, maps, config):
    """Soft JS consistency; no argmax routing, gradients reach both heads."""
    cfg = config if isinstance(config, HierarchyConfig) else HierarchyConfig(**config)
    p2, p12, p1 = [torch.softmax(x.float(), 1) for x in (logits2, logits12, logits1)]
    a = aggregation_matrix(maps['l2_to_l12'], p12.shape[1], p2.device)
    b = aggregation_matrix(maps['l12_to_l1'], p1.shape[1], p2.device)
    if a.shape[0] != p2.shape[1] or b.shape[0] != p12.shape[1]:
        raise ValueError('Hierarchy dimensions do not match checkpoint head order')
    q12, q1 = p2 @ a, p12 @ b
    terms = {'l2_l12_nll': F.nll_loss(q12.clamp_min(1e-7).log(), y12),
             'l12_l1_nll': F.nll_loss(q1.clamp_min(1e-7).log(), y1),
             'l2_l12_js': _js(p12, q12), 'l12_l1_js': _js(p1, q1)}
    return sum(getattr(cfg, k)*v for k, v in terms.items()), terms
