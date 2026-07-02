# -*- coding: utf-8 -*-
"""umizuri_catch.csv の探索的集計(v2: ゼロ釣果レコード対応)。"""
import sys
import pandas as pd

def main(path):
    df = pd.read_csv(path)
    # ゼロ釣果(釣り人なし)行と実釣果行を分ける
    zero = df[df["species"].isna()]
    caught = df[df["species"].notna()].copy()
    caught["count_alloc"] = caught["count"] / caught["n_points_shared"].clip(lower=1)

    print("="*60)
    print(f"file: {path}")
    print(f"total rows: {len(df):,}  (catch rows: {len(caught):,}, zero-catch slots: {len(zero):,})")
    print(f"days: {df['date'].nunique():,}")
    print(f"period: {df['date'].min()} .. {df['date'].max()}")
    print(f"species: {caught['species'].nunique()}")
    print(f"points: {sorted(caught['point'].dropna().unique())}")
    print()
    print("― 魚種別 出現日数 上位20 ―")
    print(caught.groupby("species")["date"].nunique().sort_values(ascending=False).head(20).to_string())
    print()
    print("― ポイント別 のべ出現回数 上位 ―")
    print(caught["point"].value_counts().to_string())
    print()
    print("― 時間帯別 のべ出現回数 ―")
    print(caught["time_slot"].value_counts().to_string())
    print()
    print("― 潮回り別 のべ出現回数 ―")
    print(caught["tide_type"].value_counts().to_string())
    print()
    target = "アジ"
    sub = caught[caught["species"] == target].copy()
    if len(sub):
        sub["water_bin"] = pd.cut(sub["water_temp"], bins=[0,12,15,18,21,24,27,40])
        print(f"― {target}: 水温帯 × 潮回り の出現日数 ―")
        print(sub.groupby(["water_bin","tide_type"], observed=True)["date"].nunique().unstack(fill_value=0).to_string())
    print("\nOK")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "umizuri_catch.csv")
