Put `pytorch_model.bin` here.

Weights are **not** in this repo (~2.6 GiB). [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) is gated: accept the license on the model card, then download with a Hugging Face token.

## 2026.10.25

Expected file:

`code/UNI2/assets/ckpts/uni2-h/pytorch_model.bin`

```python
from huggingface_hub import login, hf_hub_download

login()   # paste a token that has access to MahmoodLab/UNI2-h

path = hf_hub_download(
    repo_id="MahmoodLab/UNI2-h",
    filename="pytorch_model.bin",
    local_dir="code/UNI2/assets/ckpts/uni2-h",
)
print(path)
```

Or copy from the Hugging Face cache after a successful download:

```text
~/.cache/huggingface/hub/models--MahmoodLab--UNI2-h/snapshots/<rev>/pytorch_model.bin
```

`Image_feature_extraction.py --method UNI2` loads this path. A file smaller than 1 MB is treated as incomplete (often a failed gated download or a dangling symlink).
