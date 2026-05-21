"""Shared corpus helpers for the from-scratch track (Track A).

Stage 00 (`00_data_prep/`) writes the cleaned corpus and the split; stages 01+
(tokenizer, model) read it back through here so the paths and the structural-
tag list live in exactly one place.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_CN_DIR = DATA_DIR / "raw" / "cn"
CLEAN_DIR = DATA_DIR / "clean"
CLEAN_CN_DIR = CLEAN_DIR / "cn"
SPLITS_FILE = CLEAN_DIR / "splits.json"

# Lore-structural tags emitted by the parser. Sub-step 1 of stage 00 keeps
# these verbatim (engine markup is stripped instead); stage 01 registers each
# as one reserved special token so BPE never splits or merges it.
STORY_TAGS = ("活动名称", "章节", "章节名称", "章节简介", "正文")
OPERATOR_TAGS = (
    "干员信息", "干员名称", "干员招聘文本", "干员语音",
    "干员档案", "干员模组", "干员皮肤", "模组名称", "模组描述",
)
STRUCTURAL_TAGS = STORY_TAGS + OPERATOR_TAGS


def structural_tag_tokens():
    """Open + close form of every structural tag — e.g. '<正文>', '</正文>'."""
    tokens = []
    for tag in STRUCTURAL_TAGS:
        tokens.append(f"<{tag}>")
        tokens.append(f"</{tag}>")
    return tokens


def load_splits():
    """Parse data/clean/splits.json (written by 00_data_prep/apply_split.py)."""
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"{SPLITS_FILE} not found — run 00_data_prep/apply_split.py first."
        )
    return json.loads(SPLITS_FILE.read_text(encoding="utf-8"))


def split_files(split_name):
    """Absolute paths of every cleaned file in a flat split ('train'/'val'/…).

    Test set T0 is a per-file *chapter* selection, not a flat file list — read
    splits.json['test_t0'] directly for it.
    """
    splits = load_splits()
    if split_name not in splits:
        raise KeyError(f"unknown split '{split_name}'; have {sorted(splits)}")
    entry = splits[split_name]
    if not isinstance(entry, list):
        raise TypeError(
            f"split '{split_name}' is not a flat file list — read it directly"
        )
    return [CLEAN_DIR / rel for rel in entry]
