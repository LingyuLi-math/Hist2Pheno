

######################################################
# 2026.03.23 For model training and validation: LLY Adjust model to use level1 and level2
######################################################
import os
import numpy as np
import torch
import torch.nn as nn

from base import ImprovedMLPClassifier, evaluate

## Helper utilities shared by training/validation modes.
def infer_input_dim_and_num_classes(
    X_train_scaled,
    y_train_encoded,
    y_test_encoded,
    class_names=None,
):
    """Infer model input dimension and class count."""
    input_dim = X_train_scaled.shape[1]
    if class_names is not None:
        num_classes = len(class_names)
    else:
        # Use full label space to avoid out-of-bounds targets.
        num_classes = int(np.max(np.concatenate([y_train_encoded, y_test_encoded]))) + 1
    return input_dim, num_classes


def build_mlp_classifier(
    input_dim,
    num_classes,
    device,
    hidden_dims=(1024, 512, 256),
    dropout=0.2,
):
    """Create the MLP classifier on the given device."""
    return ImprovedMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=list(hidden_dims),
        dropout=dropout,
    ).to(device)


def get_best_checkpoint_path(path, therapy_data, therapy_model=None):
    """Return the checkpoint path saved during training."""
    if therapy_model is None:
        # Default naming used across notebooks.
        therapy_model = f"{therapy_data}_project_tumor1"
    return (
        f"{path}Collaborate/esccAI/data/{therapy_data}/"
        f"{therapy_model}/best_mlp_gpu.pt"
    )


## 训练中途变成 nan
def train_and_save_model(
    device,
    X_train_scaled,
    y_train_encoded,
    y_test_encoded,
    y_train_level1_encoded,
    y_encoded_f,
    y_level1_encoded_f,
    train_loader,
    val_loader,
    evaluate,
    save_bestmodel_path,
    class_names=None,
    val_loader_eval=None,
    hce_lambda=0.5,
    patience=10,
    max_epochs=50,
    hidden_dims=(1024, 512, 256),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
):
    import time
    import torch.nn as nn
    import torch.nn.functional as F

    class HierarchicalCrossEntropyLoss(nn.Module):
        """
        HCE = CE(level2) + lambda * CE(level1-from-level2-logits)

        level1 logits are computed by aggregating level2 probabilities
        through a child-to-parent mapping matrix.
        """
        def __init__(self, child_to_parent, num_level1_classes, class_weights_level2=None, lambda_level1=0.5):
            super().__init__()
            self.lambda_level1 = float(lambda_level1)
            if class_weights_level2 is not None:
                self.register_buffer("class_weights_level2", class_weights_level2)
            else:
                self.class_weights_level2 = None

            # map_l2_to_l1[k, p] = 1 if level2 class k belongs to level1 class p
            map_l2_to_l1 = torch.zeros(len(child_to_parent), num_level1_classes, dtype=torch.float32)
            for k, p in enumerate(child_to_parent):
                map_l2_to_l1[k, int(p)] = 1.0
            self.register_buffer("map_l2_to_l1", map_l2_to_l1)

        def forward(self, logits_l2, target_l2, target_l1):
            # Keep HCE math in fp32 for numerical stability (especially with AMP).
            logits_fp32 = torch.nan_to_num(logits_l2.float(), nan=0.0, posinf=30.0, neginf=-30.0)

            # Standard CE on level2 labels.
            ce_l2 = F.cross_entropy(logits_fp32, target_l2, weight=self.class_weights_level2)

            # Aggregate child probabilities to parent probabilities.
            probs_l2 = torch.softmax(logits_fp32, dim=1)
            probs_l1 = torch.matmul(probs_l2, self.map_l2_to_l1)
            log_probs_l1 = torch.log(probs_l1.clamp_min(1e-6))
            ce_l1 = F.nll_loss(log_probs_l1, target_l1)

            total_loss = ce_l2 + self.lambda_level1 * ce_l1
            return total_loss, ce_l2.detach(), ce_l1.detach()

    # Create the parent directory safely (dirname() can be empty).
    save_dir = os.path.dirname(save_bestmodel_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Use AMP only on CUDA.
    amp_scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    # Infer input dimension and class count.
    input_dim, num_classes = infer_input_dim_and_num_classes(
        X_train_scaled,
        y_train_encoded,
        y_test_encoded,
        class_names=class_names,
    )
    num_level1_classes = len(np.unique(y_train_level1_encoded))

    print("Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of level2 classes: {num_classes}")
    print(f"  Number of level1 classes: {num_level1_classes}")

    # Build model before creating optimizer/scheduler.
    model = build_mlp_classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    print(f"Model initialized on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer + scheduler.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Class weights for imbalanced level2 classes.
    unique_classes = np.unique(y_train_encoded)
    from sklearn.utils.class_weight import compute_class_weight

    cw = compute_class_weight("balanced", classes=unique_classes, y=y_train_encoded)
    full_weights = np.ones(num_classes, dtype=np.float64)
    full_weights[unique_classes.astype(int)] = cw
    class_weights_tensor = torch.as_tensor(full_weights, dtype=torch.float32, device=device)

    # Build level2 -> level1 mapping from the filtered dataset.
    child_to_parent = np.full(num_classes, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        if child_to_parent[int(l2)] == -1:
            child_to_parent[int(l2)] = int(l1)
        elif child_to_parent[int(l2)] != int(l1):
            raise ValueError(f"Inconsistent hierarchy: level2 class {l2} maps to multiple level1 classes")
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes: {missing}")

    criterion = HierarchicalCrossEntropyLoss(
        child_to_parent=child_to_parent,
        num_level1_classes=num_level1_classes,
        class_weights_level2=class_weights_tensor,
        lambda_level1=hce_lambda,
    ).to(device)

    print(f"HCE enabled: total_loss = CE(level2) + {hce_lambda:.3f} * CE(level1)")

    from sklearn.metrics import accuracy_score, f1_score

    def evaluate_level1_from_level2_logits(model, loader, device, child_to_parent_arr, num_l1):
        model.eval()
        all_preds_l1, all_labels_l1 = [], []

        map_l2_to_l1 = torch.zeros(len(child_to_parent_arr), num_l1, dtype=torch.float32, device=device)
        for k, p in enumerate(child_to_parent_arr):
            map_l2_to_l1[k, int(p)] = 1.0

        with torch.no_grad():
            for x, _, y_l1 in loader:
                x = x.to(device, non_blocking=True)
                y_l1 = y_l1.to(device, non_blocking=True)

                logits_l2 = model(x).float()
                probs_l2 = torch.softmax(logits_l2, dim=1)
                probs_l1 = torch.matmul(probs_l2, map_l2_to_l1)
                preds_l1 = torch.argmax(probs_l1, dim=1)

                all_preds_l1.append(preds_l1.cpu().numpy())
                all_labels_l1.append(y_l1.cpu().numpy())

        preds = np.concatenate(all_preds_l1)
        labels = np.concatenate(all_labels_l1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return acc, macro_f1, weighted_f1, preds, labels

    # Train with hierarchical targets.
    best_weighted_f1 = 0.0
    best_macro_f1 = 0.0
    best_epoch = 0
    counter = 0

    eval_loader = val_loader_eval if val_loader_eval is not None else val_loader

    print("Starting training with HCE...")
    print("=" * 60)

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_ce_l2 = 0.0
        total_ce_l1 = 0.0
        num_batches = 0

        for x, y_l2, y_l1 in train_loader:
            x = x.to(device, non_blocking=True)
            y_l2 = y_l2.to(device, non_blocking=True)
            y_l1 = y_l1.to(device, non_blocking=True)

            optimizer.zero_grad()

            if amp_scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss, ce_l2, ce_l1 = criterion(logits, y_l2, y_l1)
                if not torch.isfinite(loss):
                    print("  Warning: non-finite loss encountered (AMP), skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                logits = model(x)
                loss, ce_l2, ce_l1 = criterion(logits, y_l2, y_l1)
                if not torch.isfinite(loss):
                    print("  Warning: non-finite loss encountered, skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += float(loss.item())
            total_ce_l2 += float(ce_l2.item())
            total_ce_l1 += float(ce_l1.item())
            num_batches += 1

        # Validation metrics for level2 (original) and level1 (hierarchical aggregation).
        val_acc, val_macro_f1, val_weighted_f1, _, _ = evaluate(model, eval_loader, device, amp_scaler)
        val_l1_acc, val_l1_macro_f1, val_l1_weighted_f1, _, _ = evaluate_level1_from_level2_logits(
            model, val_loader, device, child_to_parent, num_level1_classes
        )

        avg_loss = total_loss / max(num_batches, 1)
        avg_ce_l2 = total_ce_l2 / max(num_batches, 1)
        avg_ce_l1 = total_ce_l1 / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{max_epochs} ({epoch_time:.2f}s, LR: {current_lr:.6f})")
        print(f"  Train HCE Loss: {avg_loss:.4f}")
        print(f"  CE(level2): {avg_ce_l2:.4f}, CE(level1): {avg_ce_l1:.4f}")
        print(f"  Val L2 Accuracy: {val_acc:.4f}")
        print(f"  Val L2 Macro-F1: {val_macro_f1:.4f}")
        print(f"  Val L2 Weighted-F1: {val_weighted_f1:.4f}")
        print(f"  Val L1 Accuracy: {val_l1_acc:.4f}")
        print(f"  Val L1 Macro-F1: {val_l1_macro_f1:.4f}")
        print(f"  Val L1 Weighted-F1: {val_l1_weighted_f1:.4f}")

        if val_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = val_weighted_f1
            best_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_bestmodel_path)
            counter = 0
            print(
                f"  ✓ Best model saved (Weighted-F1: {best_weighted_f1:.4f}, Macro-F1: {best_macro_f1:.4f})"
            )
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    print("=" * 60)
    print("Training completed!")
    print(f"Best validation Weighted-F1: {best_weighted_f1:.4f} at epoch {best_epoch}")
    print(f"Best validation Macro-F1: {best_macro_f1:.4f} at epoch {best_epoch}")

    return best_weighted_f1, best_macro_f1, best_epoch



######################################################
# 2026.03.23 For model validation: LLY Adjust model to use level1 and level2
######################################################
# Override mode_validation: report both level2 and level1 metrics.
def mode_validation(
    val_loader,
    device,
    path,
    therapy_data,
    therapy_model=None,
    X_train_scaled,
    y_train_encoded,
    y_test_encoded,
    y_encoded_f=None,
    y_level1_encoded_f=None,
    class_names=None,
    class_names_level1=None,
    val_loader_eval=None,
    hidden_dims=(1024, 512, 256),
    dropout=0.2,
):
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    input_dim, num_classes = infer_input_dim_and_num_classes(
        X_train_scaled,
        y_train_encoded,
        y_test_encoded,
        class_names=class_names,
    )

    model = build_mlp_classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    if y_encoded_f is None or y_level1_encoded_f is None:
        raise ValueError("y_encoded_f and y_level1_encoded_f are required.")

    checkpoint_path = get_best_checkpoint_path(path, therapy_data, therapy_model=therapy_model)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Level2 validation
    eval_loader = val_loader_eval if val_loader_eval is not None else val_loader
    val_acc, val_macro_f1, val_weighted_f1, val_preds, val_labels = evaluate(model, eval_loader, device)

    print("\nValidation Performance (Level2, best checkpoint):")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Macro F1-score: {val_macro_f1:.4f}")
    print(f"  Weighted F1-score: {val_weighted_f1:.4f}")

    unique_labels = np.unique(np.concatenate([val_labels, val_preds]))
    actual_class_names = [class_names[i] for i in unique_labels] if class_names is not None else [str(i) for i in unique_labels]
    print("\nDetailed Classification Report (Level2):")
    print(classification_report(val_labels, val_preds, labels=unique_labels, target_names=actual_class_names, zero_division=0))

    # Build level2->level1 mapping
    child_to_parent = np.full(num_classes, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        if child_to_parent[int(l2)] == -1:
            child_to_parent[int(l2)] = int(l1)
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes in validation: {missing}")

    num_level1_classes = int(np.max(child_to_parent)) + 1
    map_l2_to_l1 = torch.zeros(num_classes, num_level1_classes, dtype=torch.float32, device=device)
    for k, p in enumerate(child_to_parent):
        map_l2_to_l1[k, int(p)] = 1.0

    # Level1 validation by aggregating level2 probabilities
    model.eval()
    val_preds_level1_list, val_labels_level1_list = [], []
    with torch.no_grad():
        for x, _, y_l1 in val_loader:
            x = x.to(device, non_blocking=True)
            y_l1 = y_l1.to(device, non_blocking=True)

            logits_l2 = model(x).float()
            probs_l2 = torch.softmax(logits_l2, dim=1)
            probs_l1 = torch.matmul(probs_l2, map_l2_to_l1)
            preds_l1 = torch.argmax(probs_l1, dim=1)

            val_preds_level1_list.append(preds_l1.cpu().numpy())
            val_labels_level1_list.append(y_l1.cpu().numpy())

    val_preds_level1 = np.concatenate(val_preds_level1_list)
    val_labels_level1 = np.concatenate(val_labels_level1_list)

    val_level1_acc = accuracy_score(val_labels_level1, val_preds_level1)
    val_level1_macro_f1 = f1_score(val_labels_level1, val_preds_level1, average="macro", zero_division=0)
    val_level1_weighted_f1 = f1_score(val_labels_level1, val_preds_level1, average="weighted", zero_division=0)

    print("\nValidation Performance (Level1):")
    print(f"  Accuracy: {val_level1_acc:.4f}")
    print(f"  Macro F1-score: {val_level1_macro_f1:.4f}")
    print(f"  Weighted F1-score: {val_level1_weighted_f1:.4f}")

    unique_labels_l1 = np.unique(np.concatenate([val_labels_level1, val_preds_level1]))
    actual_class_names_l1 = [class_names_level1[i] for i in unique_labels_l1] if class_names_level1 is not None else [str(i) for i in unique_labels_l1]
    print("\nDetailed Classification Report (Level1):")
    print(classification_report(val_labels_level1, val_preds_level1, labels=unique_labels_l1, target_names=actual_class_names_l1, zero_division=0))

    return (
        model,
        input_dim,
        val_acc,
        val_macro_f1,
        val_weighted_f1,
        val_preds,
        val_labels,
        val_level1_acc,
        val_level1_macro_f1,
        val_level1_weighted_f1,
        val_preds_level1,
        val_labels_level1,
    )