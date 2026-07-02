# -*- coding: utf-8 -*-
"""
福岡市海づり公園「最近の釣果」スクレイパー v2

parser.py(v2) と組み合わせ、p184=1..426 を巡回してCSV化する。
本文抽出は <article> のテキストを取り出す方式(Claude in Chrome の
get_page_text と同じ結果になる)。パーサーはそのテキストを処理する。

使い方:
    python3 scrape.py --max 3      # 動作確認(先頭3ページ)
    python3 scrape.py              # 全426ページ
    再実行時は html_cache/ を使い再取得しない(サーバ負荷ゼロで再解析可)
"""
import argparse, csv, os, sys, time
import requests
from bs4 import BeautifulSoup
from parser import parse_page

BASE = "https://umizuri.com/pages/30/p184={n}#block184"
CACHE_DIR = "html_cache"
OUT_CSV = "umizuri_catch.csv"
MAX_PAGE = 426
HEADERS = {"User-Agent": "umizuri-research-scraper/1.0 (academic use; contact: honda.satoshi.329@m.kyushu-u.ac.jp)"}
FIELDS = ["date","tide_type","high_tides","low_tides","time_slot","water_temp","air_temp",
          "wind_dir","wind_speed","species","size_min","size_max","size_unit","count","point",
          "n_points_shared","src_page"]

def fetch_html(n, delay, session):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"page_{n:03d}.html")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f: return f.read()
    url = BASE.format(n=n)
    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            with open(cache, "w", encoding="utf-8") as f: f.write(r.text)
            time.sleep(delay)
            return r.text
        except requests.RequestException as e:
            wait = delay * (attempt + 2)
            print(f"  [warn] page {n} attempt {attempt+1}: {e}; retry in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch page {n}")

def html_to_text(html):
    """get_page_text と同等: article(なければbody)のテキストを改行区切りで取得。"""
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find("article") or soup.body or soup
    return node.get_text("\n", strip=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=MAX_PAGE)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    session = requests.Session()
    all_rows, seen_dates = [], set()
    for n in range(args.start, args.max + 1):
        print(f"[{n}/{args.max}] fetching...", file=sys.stderr)
        try:
            html = fetch_html(n, args.delay, session)
        except RuntimeError as e:
            print(f"  [error] {e}; stopping.", file=sys.stderr); break
        text = html_to_text(html)
        rows = parse_page(text)
        for r in rows:
            r["src_page"] = n
            seen_dates.add(r["date"])
        all_rows.extend(rows)
        print(f"  parsed {len(rows)} rows (cum {len(all_rows)}, {len(seen_dates)} days)", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in all_rows: w.writerow(r)
    print(f"\nWROTE {len(all_rows)} rows -> {args.out}; days={len(seen_dates)}", file=sys.stderr)

if __name__ == "__main__":
    main()
