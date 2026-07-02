# -*- coding: utf-8 -*-
"""
魚種別 在不在予測モデル(LightGBM二値分類)。

「明日、各時間帯・各ポイントで対象魚が釣れる確率」を出す。
時間で区切って学習・評価する(未来を見て過去を当てる不正を防ぐ)。

負例の作り方(重要):
  釣果記録は「釣れた」正例しか無いので、各(日, 時間帯)で実際に釣りが
  行われた(=何か記録がある)場を母集団とし、その日その時間帯に稼働
  していたポイントのうち、対象魚の記録が無い(ポイント)を負例とする。
  「稼働ポイント」= その(日,時間帯)に何らかの魚種記録があったポイント。
  これで『条件が揃った場に対象魚がいたか』を学習する。

使い方:
    python3 train_model.py umizuri_clean.csv クロ
    python3 train_model.py umizuri_clean.csv "マメアジ*"

出力:
    - 標準出力に評価指標(AUC, 適合率/再現率)と特徴量重要度
    - model_<species>.txt にモデルを保存
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

FEATURES_NUM = ["water_temp","air_temp","wind_speed","center_h","month",
                "doy_sin","doy_cos","to_high_h","to_low_h","moon_age",
                "near_sunrise","near_sunset","onshore","onshore_strength",
                "exposure","corner"]
FEATURES_CAT = ["tide_type","time_slot","point","zone","wind_dir"]

def build_dataset(df, target):
    """在不在ラベル付きデータを構成する。1行=(日,時間帯,ポイント)。"""
    # 稼働ポイント: (date,time_slot,point)ごとに、その場に何か記録があったか
    # まず実釣果行(species非空)だけで「稼働」を定義
    caught = df[df["species"].notna()].copy()

    # (date,time_slot,point) をユニーク化 = 稼働した場
    active = caught[["date","time_slot","point"]].drop_duplicates()

    # 対象魚がいた (date,time_slot,point)
    pos = caught[caught["species"] == target][["date","time_slot","point"]].drop_duplicates()
    pos["label"] = 1

    # 稼働場に正例ラベルを結合、無いものは負例(0)
    data = active.merge(pos, on=["date","time_slot","point"], how="left")
    data["label"] = data["label"].fillna(0).astype(int)

    # 各場の環境・特徴量を1件取得して結合(場ごとに環境は共通のはず)
    # FEATURES_CAT に time_slot/point が含まれ結合キーと重複するため、
    # 特徴量列からキー列を除いてから結合する。
    key = ["date","time_slot","point"]
    feat_cols = [c for c in (FEATURES_NUM + FEATURES_CAT) if c not in key]
    env = caught[key + feat_cols].drop_duplicates(subset=key)
    data = data.merge(env, on=key, how="left")
    return data

def time_split(data, cutoff):
    """cutoff日付より前=学習, 以降=評価。"""
    tr = data[data["date"] < cutoff].copy()
    te = data[data["date"] >= cutoff].copy()
    return tr, te

def main(inp, target, cutoff=None):
    df = pd.read_csv(inp)
    data = build_dataset(df, target)
    n_pos = int(data["label"].sum())
    print(f"対象魚: {target}")
    print(f"全稼働場: {len(data):,}  正例(いた): {n_pos:,}  出現率: {n_pos/len(data):.1%}")
    if n_pos < 30:
        print("正例が少なすぎます。別の魚種を選ぶか期間を広げてください。")
        return

    # 時間分割: 指定が無ければ最後の約20%期間を評価に
    dates = np.sort(data["date"].unique())
    if cutoff is None:
        cutoff = dates[int(len(dates)*0.8)]
    tr, te = time_split(data, cutoff)
    print(f"学習期間: {tr['date'].min()} .. {tr['date'].max()} ({len(tr):,}件, 正例{int(tr['label'].sum())})")
    print(f"評価期間: {te['date'].min()} .. {te['date'].max()} ({len(te):,}件, 正例{int(te['label'].sum())})")
    print()

    # カテゴリ変数をcategory型に
    for c in FEATURES_CAT:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    feats = FEATURES_NUM + FEATURES_CAT

    # クラス不均衡に対応(正例が少ないので重み付け)
    spw = (len(tr)-tr["label"].sum())/max(tr["label"].sum(),1)
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=31, min_child_samples=30, verbose=-1,
                  scale_pos_weight=spw)
    dtr = lgb.Dataset(tr[feats], tr["label"], categorical_feature=FEATURES_CAT)
    model = lgb.train(params, dtr, num_boost_round=300)

    # 評価
    pred = model.predict(te[feats])
    auc = roc_auc_score(te["label"], pred)
    ap = average_precision_score(te["label"], pred)
    print(f"=== 評価(未来データ) ===")
    print(f"AUC = {auc:.3f}  (0.5=でたらめ, 1.0=完璧)")
    print(f"PR-AUC = {ap:.3f}  (出現率={te['label'].mean():.1%} が下限の目安)")
    print()
    # 閾値0.5での分類レポート
    print("しきい値0.5での成績:")
    print(classification_report(te["label"], (pred>=0.5).astype(int),
                                target_names=["いない","いる"], digits=3, zero_division=0))

    # 特徴量重要度
    imp = pd.DataFrame({"feature":feats, "importance":model.feature_importance()})
    imp = imp.sort_values("importance", ascending=False)
    print("特徴量の効き具合(上位10):")
    print(imp.head(10).to_string(index=False))

    # 日本語ファイル名はWindowsのLightGBMが書き込めないため、
    # 魚種名をハッシュした英数字ファイル名にする(予報側も同じ変換)。
    import hashlib
    tag = hashlib.md5(target.encode('utf-8')).hexdigest()[:8]
    out = f"model_{tag}.txt"
    model.save_model(out)
    print(f"(モデルファイル: {out} = {target})")
    print(f"\nモデル保存: {out}")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv)>1 else "umizuri_clean.csv"
    target = sys.argv[2] if len(sys.argv)>2 else "クロ"
    main(inp, target)
