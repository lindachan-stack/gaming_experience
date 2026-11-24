# -*- coding: utf-8 -*-
"""實驗的變體（feel_engine/variants.py）：加標籤、輸出路徑（合成資料，不讀快取）。"""
from __future__ import annotations

import pandas as pd
import polars as pl

import feel_engine.paths as paths
from feel_engine import variants


def test_attach_tags_takes_latest_before_decision_week(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "TAG_ROOT", tmp_path)
    folder = paths.tag_dir("TCG", "PHP", "22071")
    folder.mkdir(parents=True)
    pl.DataFrame({"member_account": ["a", "b", "b"], "week_index": [202637, 202636, 202637],
                  "bet_points_tag": ["高", "低", "本週無下注"]}).write_parquet(folder / "apply_week=2026-09-14.parquet")
    cfg = {"data": {"merchant": "TCG", "currency": "PHP", "game": "22071"}}
    df = pd.DataFrame({"member_account": ["b", "a", "c"], "week_index": [202638, 202638, 202638], "x": [1, 2, 3]})
    out = variants.attach_tags(df, cfg, ["bet_points_tag"])
    assert out["member_account"].tolist() == ["b", "a", "c"]                  # 列順序不變
    assert out["TAG_bet_points"].tolist() == ["本週無下注", "高", "無紀錄"]      # 最新一筆；沒有紀錄 → 無紀錄


def test_variant_paths_without_suffix():
    vp = paths.VariantPaths("TCG_PHP_22071/train_a_valid_b", variant="")
    assert vp.experiment_html("all", "alltags").name == "all_tag_alltags.html"
    assert vp.model_dir("all", "alltags").name == "all_tag_alltags"
    assert all(v["suffix"] or v["tags"] not in ("none", "lifecycle", "value", "lifecycle_value")
               for v in variants.VARIANTS.values())                           # 沒有後綴的變體不會與正式實驗撞名
