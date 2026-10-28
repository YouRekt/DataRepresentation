"""Download, verify and load the SMS Spam Collection v.1 (UCI id 228)."""

import hashlib
import io
import tomllib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"


def load_config() -> dict:
    with open(ROOT / "configs" / "data.toml", "rb") as f:
        return tomllib.load(f)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(cfg: dict | None = None) -> Path:
    """Fetch the UCI archive once and refuse any file whose checksum differs."""
    src = (cfg or load_config())["source"]
    path = RAW_DIR / src["member"]
    if path.exists() and sha256(path.read_bytes()) == src["file_sha256"]:
        return path
    with urllib.request.urlopen(src["url"]) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        data = zf.read(src["member"])
        readme = zf.read("readme")
    if sha256(data) != src["file_sha256"]:
        raise ValueError(f"{src['member']} checksum mismatch: {sha256(data)}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    (RAW_DIR / "readme").write_bytes(readme)
    return path


def load_raw() -> pd.DataFrame:
    """One row per line of the original file; `id` is the 0-based line index.

    Text is kept byte-for-byte (no strip, no unescaping) so B1 sees the original.
    """
    lines = download().read_text(encoding="utf-8").split("\n")
    rows = [line.split("\t", 1) for line in lines if line]
    df = pd.DataFrame(rows, columns=["label", "text"])
    df.insert(0, "id", range(len(df)))
    return df


def load_split(seed: int = 42) -> pd.DataFrame:
    """Raw messages joined with the frozen split assignment."""
    split = pd.read_csv(SPLIT_DIR / f"split_seed{seed}.csv")
    df = load_raw().merge(split.drop(columns="label"), on="id", validate="1:1")
    text_sha1 = df.text.map(lambda t: hashlib.sha1(t.encode()).hexdigest())
    if not (text_sha1 == df.text_sha1).all():
        raise ValueError("raw text does not match the frozen split")
    return df
