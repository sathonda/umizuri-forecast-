# -*- coding: utf-8 -*-
"""
自動サイト生成のまとめ役(GitHub Actions から毎日呼ばれる)。

やること(全自動):
  1. データ更新   … 初回は全期間スクレイプ、2回目以降は差分更新(update.py)
  2. 特徴量→前処理 … features.py → preprocess.py
  3. 学習         … train_all.py(11魚種)
  4. 週間予報生成 … predict_week.py(本日から16日先まで)
  5. サイト用に配置 … 生成HTMLを site/index.html にコピー

出力: site/ フォルダ(GitHub Pages がこれを公開する)
     ブラウザで URL を開くだけで、日付・魚種・時間帯を選んで予報が見られる。

ローカルでも動作確認できる:
    python build_site.py
"""
import os, sys, glob, shutil, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
CATCH = os.path.join(HERE, "umizuri_catch.csv")

def run(cmd):
    """サブプロセスを実行し、失敗したら分かりやすく止める。"""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"[error] コマンド失敗: {' '.join(cmd)} (code {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)

def main():
    py = sys.executable  # 実行中の python をそのまま使う

    # 1. データ更新: CSV が無ければ初回フルスクレイプ、あれば差分更新
    if not os.path.exists(CATCH):
        print("=== 初回: 全期間を収集します(時間がかかります) ===")
        run([py, "scrape.py", "--delay", "3"])
    else:
        print("=== 差分更新します ===")
        # update.py は失敗しても(新規日ゼロ等)続行したいので個別に扱う
        r = subprocess.run([py, "update.py", "--pages", "5"], cwd=HERE)
        if r.returncode != 0:
            print("[warn] 差分更新でエラー。既存データで続行します。", file=sys.stderr)

    # 2. 特徴量 → 前処理
    run([py, "features.py", "umizuri_catch.csv", "umizuri_features.csv"])
    run([py, "preprocess.py", "umizuri_features.csv", "umizuri_clean.csv"])

    # 3. 学習(古いモデルを消してから作り直す = TARGETS変更にも安全)
    for f in glob.glob(os.path.join(HERE, "model_*.txt")):
        os.remove(f)
    run([py, "train_all.py", "umizuri_clean.csv"])

    # 4. 週間予報(本日起点16日)を生成
    #    古い forecast_week_*.html が残っていると取り違えるので、先に消す。
    for f in glob.glob(os.path.join(HERE, "forecast_week_*.html")):
        os.remove(f)
    run([py, "predict_week.py", "umizuri_clean.csv"])

    # 5. サイトへ配置(今回生成された唯一のHTMLを index.html にする)
    os.makedirs(SITE, exist_ok=True)
    weeks = sorted(glob.glob(os.path.join(HERE, "forecast_week_*.html")))
    if not weeks:
        print("[error] 週間予報HTMLが生成されませんでした。", file=sys.stderr); sys.exit(1)
    latest = weeks[-1]
    shutil.copy(latest, os.path.join(SITE, "index.html"))
    # 生成時刻を刻んだ小さな情報ファイルも置く(任意)
    with open(os.path.join(SITE, "updated.txt"), "w", encoding="utf-8") as f:
        now = datetime.datetime.now(datetime.timezone.utc)
        f.write(f"最終更新(UTC): {now:%Y-%m-%d %H:%M}\n")
    print(f"\n完成: site/index.html (元: {os.path.basename(latest)})")
    print("GitHub Pages がこの site/ フォルダを公開します。")

if __name__ == "__main__":
    main()
