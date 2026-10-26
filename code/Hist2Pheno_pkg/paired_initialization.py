########################################################
# 2026.10.26, paired ablation initialization / 同名层独立随机流，空间零初始化保持不变
########################################################
import hashlib
import random
import numpy as np
import torch
from torch import nn


def _seed(seed, fold, name):
    payload = f'hist2pheno-paired-v1:{seed}:{fold}:{name}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63-1)


def initialize_paired_model(model, seed, fold):
    """Same named shared layers start identically, independent of optional layers."""
    hashes = {}
    for name, layer in model.named_modules():
        if name.startswith('spatial_fusion'):
            continue  # 保留 projection=0、gate bias=2、geometry=0 的原始结构初始化
        if isinstance(layer, nn.Linear):
            generator = torch.Generator(device=layer.weight.device)
            generator.manual_seed(_seed(seed, fold, name))
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu', generator=generator)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for parameter_name, value in layer.named_parameters(recurse=False):
            hashes[f'{name}.{parameter_name}'] = hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    # Reset training RNG after construction / 模型构造消耗的随机数不影响训练起点
    training_seed = _seed(seed, fold, 'training')
    random.seed(training_seed)
    np.random.seed(training_seed % (2**32))
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    return {'method': 'named-layer-v1', 'seed': seed, 'fold': fold,
            'shared_parameter_sha256': hashes}
