Put `pytorch_model.bin` here (or `model.safetensors`).

Weights are **not** in this repo (~4.3 GiB). [bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) is gated and licensed **CC BY-NC-ND 4.0** (non-commercial research only). Accept the license on the model card, then download with a Hugging Face token.

## 2026.10.25

Expected file:

`code/H-optimus-1/assets/ckpts/pytorch_model.bin`

```python
from huggingface_hub import login, hf_hub_download

login()   # paste a token that has access to bioptimus/H-optimus-1

path = hf_hub_download(
    repo_id="bioptimus/H-optimus-1",
    filename="pytorch_model.bin",
    local_dir="code/H-optimus-1/assets/ckpts",
)
print(path)
```

Or copy from the Hugging Face cache after a successful download:

```text
~/.cache/huggingface/hub/models--bioptimus--H-optimus-1/snapshots/<rev>/pytorch_model.bin
```

`Image_feature_extraction.py --method H-optimus-1` loads this path (`--patch_size 224`). A file smaller than 1 MB is treated as incomplete (often a failed gated download).
