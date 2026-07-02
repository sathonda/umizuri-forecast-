# -*- coding: utf-8 -*-
"""
釣果データの差分更新スクリプト。

毎回サイト全体(426ページ・20分超)を取り直す必要はありません。
新しい釣果は「最近の釣果」の先頭ページに追加されていくので、
このスクリプトは【先頭の数ページだけを取り直し】、まだ持っていない
日付のデータだけを既存CSVに追記します。古いページのキャッシュ
(html_cache/)はそのまま使うのでサーバ負荷も時間も最小です。

なぜ先頭ページを取り直すのか:
  新しい釣果が載るとページ1の中身が変わります。scrape.py は
  ページ番号でキャッシュ(page_001.html)するため、放っておくと
  古い内容のまま再利用してしまいます。そこでこのスクリプトは、
  指定ページ数ぶんのキャッシュを一旦消してから取得し直します。

重複の扱い:
  取り直した先頭ページには、既に持っている日付も含まれます。
  既存CSVに無い日付の行だけを追記するので、二重計上は起きません。
  (同じ日が後から追記・修正された場合は下の --refresh-days を参照)

使い方:
    python update.py                      # 先頭5ページを取り直して差分追記
    python update.py --pages 8            # 取り直すページ数を増やす(長期間空けたとき)
    python update.py --refresh-days 2026-06-27,2026-06-28
                                          # 指定日を最新版で上書き更新する

出力: umizuri_catch.csv を更新(バックアップを umizuri_catch.bak.csv に自動保存)
更新後は features → preprocess → train_all を回してモデルを作り直します
(下記 RUN_STEPS.md / 説明書.md 参照)。
"""
import argparse, csv, os, sys, shutil, time
import requests
from bs4 import BeautifulSoup
from parser import parse_page

BASE = "https://umizuri.com/pages/30/p184={n}#block184"
CACHE_DIR = "html_cache"
OUT_CSV = "umizuri_catch.csv"
HEADERS = {"User-Agent": "umizuri-research-scraper/1.0 (academic use; contact: honda.satoshi.329@m.kyushu-u.ac.jp)"}
FIELDS = ["date","tide_type","high_tides","low_tides","time_slot","water_temp","air_temp",
          "wind_dir","wind_speed","species","size_min","size_max","size_unit","count","point",
          "n_points_shared","src_page"]

def fetch_fresh(n, delay, session):
    """キャッシュを無視して必ず取得し直し、キャッシュも更新する。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"page_{n:03d}.html")
    url = BASE.format(n=n)
    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            with open(cache, "w", encoding="utf-8") as f:
                f.write(r.text)
            time.sleep(delay)
            return r.text
        except requests.RequestException as e:
            wait = delay * (attempt + 2)
            print(f"  [warn] page {n} attempt {attempt+1}: {e}; retry in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch page {n}")

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find("article") or soup.body or soup
    return node.get_text("\n", strip=True)

def load_existing(path):
    rows, dates = [], set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r); dates.add(r.get("date"))
    return rows, dates

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=5,
                    help="取り直す先頭ページ数(既定5)。長く空けたら増やす")
    ap.add_argument("--delay", type=float, default=3.0, help="ページ取得間隔(秒)")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--refresh-days", default="",
                    help="この日付(カンマ区切り)は既存CSVから削除し最新版で入れ直す")
    args = ap.parse_args()

    if not os.path.exists(args.out):
        print(f"[error] {args.out} が見つかりません。初回は scrape.py で全期間を取得してください。",
              file=sys.stderr)
        sys.exit(1)

    # バックアップ
    bak = args.out.replace(".csv", ".bak.csv")
    shutil.copy(args.out, bak)
    print(f"バックアップ: {bak}", file=sys.stderr)

    existing_rows, existing_dates = load_existing(args.out)
    print(f"既存データ: {len(existing_rows):,} 行 / {len(existing_dates):,} 日", file=sys.stderr)

    refresh = {d.strip() for d in args.refresh_days.split(",") if d.strip()}
    if refresh:
        before = len(existing_rows)
        existing_rows = [r for r in existing_rows if r.get("date") not in refresh]
        existing_dates -= refresh
        print(f"再取得対象として {len(refresh)} 日を一旦削除({before-len(existing_rows)}行)", file=sys.stderr)

    # 先頭ページを取り直す(キャッシュを消してから取得)
    session = requests.Session()
    fresh_rows = []
    for n in range(1, args.pages + 1):
        cache = os.path.join(CACHE_DIR, f"page_{n:03d}.html")
        if os.path.exists(cache):
            os.remove(cache)   # 古い内容を捨てて必ず取り直す
        print(f"[{n}/{args.pages}] 取得中 ...", file=sys.stderr)
        try:
            html = fetch_fresh(n, args.delay, session)
        except RuntimeError as e:
            print(f"  [error] {e}; ここで中断", file=sys.stderr); break
        rows = parse_page(html_to_text(html))
        for r in rows:
            r["src_page"] = n
        fresh_rows.extend(rows)
        print(f"  {len(rows)} 行 解析", file=sys.stderr)

    # 既存に無い日付の行だけ追記
    new_rows = [r for r in fresh_rows if r.get("date") not in existing_dates]
    new_days = sorted({r.get("date") for r in new_rows})
    print(f"新規の日付: {len(new_days)} 日 → {new_days}", file=sys.stderr)
    print(f"追記する行数: {len(new_rows):,}", file=sys.stderr)

    all_rows = existing_rows + new_rows
    # 日付順に並べ直して書き出し(既存の並びは崩さなくても学習には影響しないが読みやすさ優先)
    all_rows.sort(key=lambda r: (r.get("date") or "", r.get("time_slot") or "", r.get("point") or ""))
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    total_days = len({r.get("date") for r in all_rows})
    print(f"\n更新完了: {len(all_rows):,} 行 / {total_days:,} 日 → {args.out}", file=sys.stderr)
    if not new_days and not refresh:
        print("(新しい日付はありませんでした。まだ次の釣果が載っていないようです)", file=sys.stderr)
    print("\n次にモデルを作り直してください:", file=sys.stderr)
    print("  python features.py umizuri_catch.csv umizuri_features.csv", file=sys.stderr)
    print("  python preprocess.py umizuri_features.csv umizuri_clean.csv", file=sys.stderr)
    print("  python train_all.py umizuri_clean.csv", file=sys.stderr)

if __name__ == "__main__":
    main()
