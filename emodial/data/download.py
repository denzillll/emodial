"""Download raw datasets into data/raw. Every step is skipped if the files already exist,
and every source can be replaced by files you download yourself (see README)."""
from __future__ import annotations

import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ED_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
ESCONV_URL = "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json"
MELD_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{split}_sent_emo.csv"
DD_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"
DD_HF = "li2017dailydialog/daily_dialog"


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "emodial-course-project"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_ed(raw: Path) -> None:
    dest = raw / "empatheticdialogues"
    if (dest / "train.csv").exists():
        return
    print("Downloading EmpatheticDialogues ...")
    with tarfile.open(fileobj=io.BytesIO(_get(ED_URL)), mode="r:gz") as tf:
        tf.extractall(raw, filter="data")


def fetch_esconv(raw: Path) -> None:
    f = raw / "ESConv.json"
    if not f.exists():
        print("Downloading ESConv ...")
        f.write_bytes(_get(ESCONV_URL))


def fetch_meld(raw: Path) -> None:
    d = raw / "meld"
    d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        f = d / f"{split}_sent_emo.csv"
        if not f.exists():
            print(f"Downloading MELD {split} ...")
            f.write_bytes(_get(MELD_URL.format(split=split)))


def fetch_dailydialog(raw: Path) -> None:
    d = raw / "dailydialog"
    if list(d.rglob("dialogues_emotion_train.txt")):
        return
    d.mkdir(parents=True, exist_ok=True)
    try:
        print("Downloading DailyDialog (original zip) ...")
        outer = zipfile.ZipFile(io.BytesIO(_get(DD_URL)))
        outer.extractall(d)
        for inner in d.rglob("*.zip"):
            zipfile.ZipFile(inner).extractall(inner.parent)
        return
    except Exception as e:  # the author's site is sometimes down
        print(f"  original source failed ({e}); trying the Hugging Face copy ...")
    from datasets import load_dataset  # noqa: WPS433

    ds = load_dataset(DD_HF, revision="refs/convert/parquet")
    for hf_split, ours in (("train", "train"), ("validation", "validation"), ("test", "test")):
        (d / ours).mkdir(exist_ok=True)
        with open(d / ours / f"dialogues_{ours}.txt", "w") as ft, open(d / ours / f"dialogues_emotion_{ours}.txt", "w") as fe:
            for row in ds[hf_split]:
                ft.write(" __eou__ ".join(u.strip() for u in row["dialog"]) + " __eou__\n")
                fe.write(" ".join(str(x) for x in row["emotion"]) + "\n")


def fetch_all(raw: str | Path) -> None:
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    for fn in (fetch_esconv, fetch_meld, fetch_ed, fetch_dailydialog):
        try:
            fn(raw)
        except Exception as e:
            print(f"!! {fn.__name__} failed: {e}\n   Download it manually into {raw} (see README, 'Data').")
