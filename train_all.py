# -*- coding: utf-8 -*-
"""
複数魚種の在不在モデルを一括学習する。
train_model.py のロジックを流用し、指定魚種すべてでモデルを作る。

使い方:
    python3 train_all.py umizuri_clean.csv

対象魚種は下の TARGETS。各魚種で model_<species>.txt を保存し、
成績(AUC等)を一覧表示する。正例が少なすぎる魚種は自動でスキップ。
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

TARGETS = ["クロ","アジ","キス","ヒラメ","セイゴ","カサゴ（アラカブ）",
           "キジハタ","メバル","バリ","マルアジ","アオリイカ"]

FEATURES_NUM = ["water_temp","air_temp","wind_speed","center_h","month",
                "doy_sin","doy_cos","to_high_h","to_low_h","moon_age",
                "near_sunrise","near_sunset","onshore","onshore_strength","exposure","corner"]
FEATURES_CAT = ["tide_type","time_slot","point","zone","wind_dir"]

def build_dataset(df, target):
    caught = df[df["species"].notna()].copy()
    active = caught[["date","time_slot","point"]].drop_duplicates()
    pos = caught[caught["species"]==target][["date","time_slot","point"]].drop_duplicates()
    pos["label"]=1
    data = active.merge(pos, on=["date","time_slot","point"], how="left")
    data["label"]=data["label"].fillna(0).astype(int)
    key=["date","time_slot","point"]
    feat_cols=[c for c in (FEATURES_NUM+FEATURES_CAT) if c not in key]
    env=caught[key+feat_cols].drop_duplicates(subset=key)
    return data.merge(env, on=key, how="left")

def train_one(df, target):
    data=build_dataset(df,target)
    n_pos=int(data["label"].sum())
    if n_pos<30:
        return dict(species=target, status="skip(正例<30)", n_pos=n_pos)
    dates=np.sort(data["date"].unique())
    cutoff=dates[int(len(dates)*0.8)]
    tr=data[data["date"]<cutoff].copy(); te=data[data["date"]>=cutoff].copy()
    if te["label"].sum()<5 or tr["label"].sum()<10:
        return dict(species=target, status="skip(評価/学習の正例不足)", n_pos=n_pos)
    for c in FEATURES_CAT:
        tr[c]=tr[c].astype("category"); te[c]=te[c].astype("category")
    feats=FEATURES_NUM+FEATURES_CAT
    spw=(len(tr)-tr["label"].sum())/max(tr["label"].sum(),1)
    params=dict(objective="binary",metric="auc",learning_rate=0.05,num_leaves=31,
                min_child_samples=30,verbose=-1,scale_pos_weight=spw)
    dtr=lgb.Dataset(tr[feats],tr["label"],categorical_feature=FEATURES_CAT)
    model=lgb.train(params,dtr,num_boost_round=300)
    pred=model.predict(te[feats])
    auc=roc_auc_score(te["label"],pred) if te["label"].nunique()>1 else float("nan")
    ap=average_precision_score(te["label"],pred) if te["label"].nunique()>1 else float("nan")
    out=f"model_{TARGETS.index(target):02d}.txt"
    model.save_model(out)
    return dict(species=target, status="ok", n_pos=n_pos,
                auc=round(auc,3), pr_auc=round(ap,3),
                base_rate=round(te["label"].mean(),3), model=out)

def main(inp):
    df=pd.read_csv(inp)
    results=[]
    for t in TARGETS:
        print(f"学習中: {t} ...", file=sys.stderr)
        results.append(train_one(df,t))
    print("\n=== 一括学習 結果 ===")
    rep=pd.DataFrame(results)
    cols=[c for c in ["species","status","n_pos","base_rate","auc","pr_auc","model"] if c in rep.columns]
    print(rep[cols].to_string(index=False))
    ok=rep[rep["status"]=="ok"] if "status" in rep else rep
    if len(ok):
        print(f"\n学習成功: {len(ok)}/{len(TARGETS)} 魚種")
        if "auc" in ok:
            print(f"AUC平均: {ok['auc'].mean():.3f}  最高: {ok['auc'].max():.3f}({ok.loc[ok['auc'].idxmax(),'species']})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "umizuri_clean.csv")
