"""2026.11.15 Slide-level clinical prediction: H&E only vs phenotype only vs both.

Independent re-implementation of the Hist2Prot clinical design (Methods,
"Clinical prediction"); no Hist2Prot code is reused (CC BY-NC-ND licence).

  pheno branch   region vectors (slide_features.py) -> Linear -> Transformer
                 encoder with a learnable CLS token -> LayerNorm -> Linear
  H&E branch     UNI2-h patch features -> gated-attention MIL (Ilse et al. 2018)
  fusion         both projected to d, concatenated, then channel attention and
                 modality attention (CBAM-style: F' = Mc(F)⊙F, F'' = Ms(F')⊙F')
  heads          binary (BCE) or survival (negative Cox partial log-likelihood)

Evaluation follows the paper: five-fold patient-level CV, every patient gets
one out-of-fold prediction, and the three settings share the same folds so
they can be compared pairwise (AUC: DeLong; C-index: paired bootstrap).

Manifest CSV columns
    patient_id, slide_id, pheno_npz, uni2_npz, and either `label` (0/1) or
    `time` + `event` (1 = event). `pheno_npz` has `region_features`; `uni2_npz`
    has `features` (n_patches × 1536). Several slides per patient are allowed:
    their bags are concatenated.

    python code/Downstream/slide_fusion.py --manifest tcga_hnsc_hpv.csv --task binary --out runs/hnsc_hpv
    python code/Downstream/slide_fusion.py --manifest tcga_hnsc_os.csv  --task survival --out runs/hnsc_os
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

SETTINGS = ("he", "pheno", "both")


# ----------------------------------------------------------------------------- model
class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim, d=256, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(dropout))
        self.a = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.b = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.w = nn.Linear(d, 1)

    def forward(self, x):                     # x: (n, in_dim)
        h = self.proj(x)
        att = torch.softmax(self.w(self.a(h) * self.b(h)).squeeze(-1), 0)
        return att @ h                        # (d,)


class PhenoAggregator(nn.Module):
    def __init__(self, in_dim, d=128, layers=2, heads=4, dropout=0.1, max_regions=4096):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        self.max_regions = max_regions

    def forward(self, r):                     # r: (n_regions, in_dim)
        if self.training and r.shape[0] > self.max_regions:
            r = r[torch.randperm(r.shape[0], device=r.device)[: self.max_regions]]
        h = torch.cat([self.cls, self.inp(r).unsqueeze(0)], 1)
        return self.out(self.norm(self.enc(h)[0, 0]))


class ModalityFusion(nn.Module):
    """Channel attention over the concatenated vector, then per-modality weights."""

    def __init__(self, d, n_mod, reduction=4):
        super().__init__()
        D = d * n_mod
        self.mc = nn.Sequential(nn.Linear(D, max(D // reduction, 8)), nn.ReLU(),
                                nn.Linear(max(D // reduction, 8), D), nn.Sigmoid())
        self.ms = nn.Linear(D, n_mod)
        self.d, self.n_mod = d, n_mod

    def forward(self, parts):
        F = torch.cat(parts, -1)
        F1 = self.mc(F) * F
        w = torch.softmax(self.ms(F1), -1)                        # (n_mod,)
        F2 = (F1.view(self.n_mod, self.d) * w.unsqueeze(-1)).reshape(-1)
        return F2, w


class SlideModel(nn.Module):
    def __init__(self, setting, he_dim=None, pheno_dim=None, d=128, dropout=0.25):
        super().__init__()
        self.setting = setting
        mods = []
        if setting in ("he", "both"):
            self.he = GatedAttentionMIL(he_dim, d, dropout); mods.append("he")
        if setting in ("pheno", "both"):
            self.ph = PhenoAggregator(pheno_dim, d); mods.append("pheno")
        self.mods = mods
        self.fuse = ModalityFusion(d, len(mods)) if len(mods) > 1 else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d * len(mods), 1))

    def forward(self, he=None, pheno=None):
        parts = []
        if "he" in self.mods:
            parts.append(self.he(he))
        if "pheno" in self.mods:
            parts.append(self.ph(pheno))
        z = self.fuse(parts)[0] if self.fuse is not None else parts[0]
        return self.head(z).squeeze(-1)


# ----------------------------------------------------------------------------- losses / metrics
def cox_nll(risk, time, event):
    """Negative Cox partial log-likelihood (Breslow ties)."""
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    log_cum = torch.logcumsumexp(r, 0)
    n_ev = e.sum()
    if n_ev == 0:
        return risk.sum() * 0.0
    return -((r - log_cum) * e).sum() / n_ev


def c_index(time, event, risk):
    time, event, risk = map(np.asarray, (time, event, risk))
    num = den = 0.0
    for i in np.flatnonzero(event == 1):
        m = time > time[i]
        den += m.sum()
        num += (risk[i] > risk[m]).sum() + 0.5 * (risk[i] == risk[m]).sum()
    return num / den if den else np.nan


def _midrank(x):
    order = np.argsort(x); xs = x[order]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n); out[order] = t
    return out


def delong_paired(y, s1, s2):
    """Two-sided paired DeLong test; returns (auc1, auc2, p)."""
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m, n = len(pos), len(neg)
    S = np.vstack([s1, s2])
    aucs, v10, v01 = [], [], []
    for s in S:
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.r_[s[pos], s[neg]])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    cov = np.cov(np.vstack(v10)) / m + np.cov(np.vstack(v01)) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z)))


def paired_bootstrap_cindex(time, event, r1, r2, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(time); d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if event[i].sum() == 0:
            continue
        d.append(c_index(time[i], event[i], r1[i]) - c_index(time[i], event[i], r2[i]))
    d = np.asarray(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]).tolist(), float(min(p, 1.0))


# ----------------------------------------------------------------------------- data
def load_bags(man):
    bags = {}
    for pid, g in man.groupby("patient_id", sort=False):
        he = [np.load(f)["features"] for f in g["uni2_npz"]] if "uni2_npz" in g else None
        ph = [np.load(f)["region_features"] for f in g["pheno_npz"]] if "pheno_npz" in g else None
        bags[pid] = (np.concatenate(he).astype(np.float32) if he else None,
                     np.concatenate(ph).astype(np.float32) if ph else None)
    return bags


def make_folds(pat, task, n_splits=5, seed=42):
    ids = pat.index.to_numpy()
    if task == "binary":
        sp = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(ids, pat["label"])
    else:
        sp = KFold(n_splits, shuffle=True, random_state=seed).split(ids)
    fold = pd.Series(-1, index=ids)
    for k, (_, te) in enumerate(sp):
        fold.iloc[te] = k
    return fold


def _standardiser(bags, ids, key):
    X = np.concatenate([bags[i][key] for i in ids])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def run_cv(man, task, out, epochs=40, lr=2e-4, wd=1e-4, d=128, seed=42, device="cpu"):
    pat = man.groupby("patient_id", sort=False).first()
    bags = load_bags(man)
    fold = make_folds(pat, task, seed=seed)
    has_he = all(bags[i][0] is not None for i in pat.index)
    has_ph = all(bags[i][1] is not None for i in pat.index)
    settings = [s for s in SETTINGS if (s != "he" or has_he) and (s != "pheno" or has_ph)
                and (s != "both" or (has_he and has_ph))]
    oof = pd.DataFrame(index=pat.index)
    oof["fold"] = fold
    for setting in settings:
        pred = pd.Series(np.nan, index=pat.index)
        for k in sorted(fold.unique()):
            tr, te = pat.index[fold != k], pat.index[fold == k]
            torch.manual_seed(seed + k); np.random.seed(seed + k)
            norm_he = _standardiser(bags, tr, 0) if setting != "pheno" else None
            norm_ph = _standardiser(bags, tr, 1) if setting != "he" else None

            def tens(pid):
                he, ph = bags[pid]
                a = torch.as_tensor((he - norm_he[0]) / norm_he[1], device=device) if norm_he else None
                b = torch.as_tensor((ph - norm_ph[0]) / norm_ph[1], device=device) if norm_ph else None
                return a, b
            model = SlideModel(setting, he_dim=bags[tr[0]][0].shape[1] if norm_he else None,
                               pheno_dim=bags[tr[0]][1].shape[1] if norm_ph else None, d=d).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            for ep in range(epochs):
                model.train()
                order = np.random.permutation(len(tr))
                if task == "binary":
                    for i in order:
                        a, b = tens(tr[i])
                        y = torch.tensor(float(pat.loc[tr[i], "label"]), device=device)
                        loss = nn.functional.binary_cross_entropy_with_logits(model(a, b), y)
                        opt.zero_grad(); loss.backward(); opt.step()
                else:
                    for chunk in np.array_split(order, max(1, len(order) // 16)):
                        risks = torch.stack([model(*tens(tr[i])) for i in chunk])
                        t = torch.as_tensor(np.array(pat.loc[tr[chunk], "time"], dtype=float), device=device)
                        e = torch.as_tensor(np.array(pat.loc[tr[chunk], "event"], dtype=float), device=device)
                        loss = cox_nll(risks, t, e)
                        opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                for pid in te:
                    pred[pid] = float(model(*tens(pid)))
        oof[setting] = pred
    out.mkdir(parents=True, exist_ok=True)
    oof.join(pat[[c for c in ("label", "time", "event") if c in pat]]).to_csv(out / "oof_predictions.csv")
    report = {"task": task, "n_patients": int(len(pat)), "settings": settings, "metrics": {}, "comparisons": {}}
    for s in settings:
        if task == "binary":
            report["metrics"][s] = float(roc_auc_score(pat["label"], oof[s]))
        else:
            report["metrics"][s] = float(c_index(pat["time"].to_numpy(), pat["event"].to_numpy(), oof[s].to_numpy()))
    for a, b in (("both", "he"), ("both", "pheno")):
        if a in settings and b in settings:
            if task == "binary":
                _, _, p = delong_paired(pat["label"].to_numpy(), oof[a].to_numpy(), oof[b].to_numpy())
                report["comparisons"][f"{a}_vs_{b}"] = {"test": "paired DeLong", "p": p}
            else:
                dlt, ci, p = paired_bootstrap_cindex(pat["time"].to_numpy(), pat["event"].to_numpy(),
                                                     oof[a].to_numpy(), oof[b].to_numpy())
                report["comparisons"][f"{a}_vs_{b}"] = {"test": "paired bootstrap", "delta": dlt, "ci95": ci, "p": p}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--task", choices=("binary", "survival"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    man = pd.read_csv(args.manifest)
    run_cv(man, args.task, args.out, epochs=args.epochs, lr=args.lr, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
