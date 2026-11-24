# -*- coding: utf-8 -*-
"""指令入口（cli.py）：讀取 dataset 的範圍（不讀快取）。"""
from __future__ import annotations

import polars as pl

from feel_engine import cli
from feel_engine.pipeline import store
from feel_engine.pipeline.data import load_config


def _spy(monkeypatch) -> list:
    reads: list = []
    monkeypatch.setattr(cli, "check_days", lambda cfg, s, days: [])
    monkeypatch.setattr(store, "load_dataset", lambda s, days, columns=None: reads.append(list(days)) or pl.DataFrame({"TAG_lifecycle": ["活躍"]}))
    return reads


def test_evaluate_reads_only_the_validation_days(monkeypatch):
    """evaluate 只用驗證期間：不把訓練期間讀進記憶體。"""
    reads = _spy(monkeypatch)
    cfg = load_config()
    train, valid = cli._load_train_valid(cfg, with_train=False)
    assert train is None and len(valid) == 1
    assert reads == [cli._run_days(cfg)[1]]


def test_run_reads_training_and_validation_days(monkeypatch):
    reads = _spy(monkeypatch)
    cfg = load_config()
    train, valid = cli._load_train_valid(cfg)
    assert len(train) == 1 and len(valid) == 1
    assert reads == list(cli._run_days(cfg))


def test_loaded_days_apply_the_config_regroup(monkeypatch):
    """run／evaluate 讀入 dataset 時套用 config.yaml 的 regroup（dataset 檔案本身不動）。"""
    monkeypatch.setattr(cli, "check_days", lambda cfg, s, days: [])
    monkeypatch.setattr(store, "load_dataset", lambda s, days, columns=None: pl.DataFrame(
        {"TAG_lifecycle": ["流失", "回流", "活躍"], "SEG_lifecycle": ["流失", "回流", "活躍"]}))
    cfg = {**load_config(), "regroup": {"TAG_lifecycle": {"流失": "回流"}}}
    train, valid = cli._load_train_valid(cfg)
    for df in (train, valid):
        assert df["TAG_lifecycle"].tolist() == ["回流", "回流", "活躍"]
        assert df["SEG_lifecycle"].tolist() == ["回流", "回流", "活躍"]
