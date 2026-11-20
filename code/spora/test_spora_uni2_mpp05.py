"""2026.11.20 spora_uni2_worker --target-mpp 0.5: archive the 1 µm store, extract with scale 2."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spora_uni2_worker as W


def _store(tmp_path, scale_image, scale):
    tif = tmp_path / "ds" / "t1" / "t1_he.tif"
    tif.parent.mkdir(parents=True)
    tif.write_bytes(b"x")
    pth = tif.parent / "t1_project_all_UNI2" / "ImgEmbeddings_all" / "sc_pth_14_14"
    pth.mkdir(parents=True)
    (pth / "embedding_manifest.json").write_text(json.dumps({"state": "completed", "scale_image": scale_image,
                                                             "scale": scale}))
    lst = tmp_path / "list.txt"
    lst.write_text(str(tif) + "\n")
    return tif, pth, lst


def test_archives_and_rescales(tmp_path, monkeypatch):
    tif, pth, lst = _store(tmp_path, False, 1.0)
    calls = []
    monkeypatch.setattr(W, "_done", lambda p: (p / "embedding_manifest.json").is_file())
    monkeypatch.setattr(W.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert W.main(["--list", str(lst), "--target-mpp", "0.5"]) == 0
    cmd = calls[0]
    assert cmd[cmd.index("--scale_image") + 1] == "True" and cmd[cmd.index("--scale") + 1] == "2"
    archived = [p.name for p in pth.parent.iterdir() if p.name.startswith("sc_pth_14_14_1mpp_archive_")]
    assert len(archived) == 1 and not pth.exists()


def test_store_at_target_scale_is_skipped(tmp_path, monkeypatch):
    tif, pth, lst = _store(tmp_path, True, 2.0)
    calls = []
    monkeypatch.setattr(W, "_done", lambda p: True)
    monkeypatch.setattr(W.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert W.main(["--list", str(lst), "--target-mpp", "0.5"]) == 0 and not calls
    # 2026.11.20 (2): the default is 0.5 µm/px too
    assert W.main(["--list", str(lst)]) == 0 and not calls


def test_default_is_half_micron(tmp_path, monkeypatch):
    # 2026.11.20 (2), no flag: the 1 µm/px store is archived and re-extracted x2
    tif, pth, lst = _store(tmp_path, False, 1.0)
    calls = []
    monkeypatch.setattr(W, "_done", lambda p: (p / "embedding_manifest.json").is_file())
    monkeypatch.setattr(W.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    W.main(["--list", str(lst)])
    assert calls[0][calls[0].index("--scale_image") + 1] == "True" and calls[0][calls[0].index("--scale") + 1] == "2"
    assert not pth.exists()
