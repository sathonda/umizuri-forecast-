# -*- coding: utf-8 -*-
"""
前処理: 名寄せ・時間帯整理・異常行除外を umizuri_features.csv に適用する。
モデル学習の直前段階。features.py の後に実行する。

適用内容(ユーザー確定方針):
  - 名寄せ: アジゴ, マメアジ → 「マメアジ*」に統合。素の「アジ」は別扱い。
            クロダイ類(チヌ/メイタ/タイゴ)は統合しない。
  - 時間帯: 主要6区分のみ残す。変則表記(7-9,15-19等)と
            異常(開始>=終了, 6-12のような3時間超)は除外。
  - ゼロ釣果行(species空)は在不在モデルの負例として保持。

使い方:
    python3 preprocess.py umizuri_features.csv umizuri_clean.csv
"""
import sys
import pandas as pd

# 統合ルール: 別名 → 代表名
SPECIES_MERGE = {
    "アジゴ": "マメアジ*",
    "マメアジ": "マメアジ*",
    # 素の「アジ」は変換しない(別扱い)
    # クロダイ類は変換しない(チヌ/メイタ/タイゴを保持)
}

# 採用する標準時間帯(中央時刻が定義できるもの)
VALID_SLOTS = {"6-9", "9-12", "12-15", "15-18", "18-20"}

def normalize_species(name):
    if pd.isna(name):
        return name
    return SPECIES_MERGE.get(name, name)

def main(inp, out):
    df = pd.read_csv(inp)
    n0 = len(df)

    # 1) 時間帯フィルタ: 標準6区分のみ。ゼロ釣果行も時間帯が標準なら残す。
    df = df[df["time_slot"].isin(VALID_SLOTS)].copy()
    n1 = len(df)

    # 2) 名寄せ
    df["species_raw"] = df["species"]           # 元名を保存(後で確認用)
    df["species"] = df["species"].map(normalize_species)

    # 3) 念のための異常値ガード: 尾数が負, サイズが異常など
    if "count" in df:
        df = df[(df["count"].isna()) | (df["count"] >= 0)]
    n2 = len(df)

    df.to_csv(out, index=False, encoding="utf-8-sig")

    # レポート
    print(f"入力 {n0:,} 行", file=sys.stderr)
    print(f"  時間帯フィルタ後: {n1:,} 行 (除外 {n0-n1:,})", file=sys.stderr)
    print(f"  異常値ガード後  : {n2:,} 行", file=sys.stderr)
    print(f"  → {out}", file=sys.stderr)
    # 名寄せの効果を表示
    merged = df[df["species"] == "マメアジ*"]
    if len(merged):
        src = merged["species_raw"].value_counts().to_dict()
        print(f"  マメアジ* に統合: {src}", file=sys.stderr)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "umizuri_features.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "umizuri_clean.csv"
    main(inp, out)
