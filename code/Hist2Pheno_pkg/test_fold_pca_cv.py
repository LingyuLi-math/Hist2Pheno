"""2026.11.20 end-to-end: grouped CV with the train-fold PCA scaler and the k=8 neighbour mean."""
import os
import tempfile

import numpy as np
import torch

import projected_scaler as ps
from test_l12_class_weight import _pancancer_cv


def test_group_cv_with_fold_pca_and_spatial(monkeypatch):
    import model as m
    from base import evaluate
    from checkpoint_artifact import read_artifact, restore_scaler
    from plot import mlp_collect_five_head_softmax_probs
    from spatial_context import SpatialConfig

    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0); np.random.seed(0)
        cv, _ = _pancancer_cv()
        d = cv["X_f"].shape[1]
        rng = np.random.default_rng(3)
        glob = (cv["X_f"] @ rng.normal(size=(d, 3))) @ rng.normal(size=(3, 10)) + 0.05 * rng.normal(size=(len(cv["X_f"]), 10))
        cv["X_f"] = np.hstack([cv["X_f"], glob]).astype(np.float16)     # pooled like uni2_tile_lg (float16)
        cv["_data_fingerprint"] = "synthetic-fold-pca-v1"
        monkeypatch.setenv(ps.ENV, f"split={d},dim=3")
        cv.update(spatial_config=vars(SpatialConfig(k=8, cache_dir=os.path.join(tmp, "ctx"))))
        out = m.train_hce_group_kfold_cv(
            device=torch.device("cpu"), cv_data=cv, evaluate=evaluate,
            class_names=list(cv["class_names"]),
            save_bestmodel_path_pattern=os.path.join(tmp, "fp_fold_{}.pt"),
            n_splits=3, hce_w12=2.0, hce_w_l3=0.0, hce_w_l4=0.0, max_epochs=2, patience=2,
            hidden_dims=(32, 16, 8), val_selection_metric="three_tier_auc_sum",
            group_splitter="cross_dataset_section",
            use_spatial_context=True, spatial_k=8, spatial_mode="mean",
            loader_kwargs={"seed": 0, "batch_size_cuda": 256, "num_workers_cuda": 0})
        assert out is not None
        ck = os.path.join(tmp, "fp_fold_0.pt")
        art = read_artifact(ck, required=True)
        scaler = restore_scaler(art["scaler"])
        assert isinstance(scaler, ps.ProjectedScaler) and scaler.n_features_in_ == d + 10
        assert scaler.n_features_out_ == d + 3
        assert m._network_input_dim(cv, art)[0] == scaler.n_features_out_
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        model, _ = m.build_mlp_classifier_for_state_dict(
            sd, input_dim=scaler.n_features_out_, num_classes=20, num_level1_classes=5,
            device=torch.device("cpu"), hidden_dims=(32, 16, 8),
            spatial_config=art["metadata"].get("spatial_config"))
        model.load_state_dict(sd); model.eval()
        model.checkpoint_scaler = scaler
        sec = cv["groups_f"] == 0
        cfg = dict(art["metadata"]["spatial_config"]); cfg["cache_dir"] = os.path.join(tmp, "ctx_inf")
        probs = mlp_collect_five_head_softmax_probs(
            model, scaler.transform(cv["X_f"][sec]),
            torch.device("cpu"), neighbor_index={"xy": cv["X_coords_f"][sec], "config_override": cfg})
        assert probs["l2"].shape == (int(sec.sum()), 20) and np.isfinite(probs["l2"]).all()
