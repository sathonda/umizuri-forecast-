# -*- coding: utf-8 -*-
"""
福岡市海づり公園「最近の釣果」パーサー v2
実ページ巡回で判明した表記パターンに対応。
"""
import re

def normalize(text: str) -> str:
    text = text.replace("：", ":")
    for ch in ["～", "〜", "∼", "‐"]:
        text = text.replace(ch, "~")
    text = text.replace("㎝", "cm").replace("ｃｍ", "cm")
    return text

TIDE_CANON = {"大汐":"大潮","大潮":"大潮","中潮":"中潮","小潮":"小潮",
              "長潮":"長潮","若潮":"若潮","若汐":"若潮"}
def canon_tide(name): return TIDE_CANON.get(name, name)

RE_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
RE_TIDE_HEAD = re.compile(r"^\s*(大汐|大潮|中潮|小潮|長潮|若潮|若汐)\b(.*)$", re.M)
RE_TIME_OR_NA = re.compile(r"(\d{1,2}:\d{2}|-{1,2}:-{1,2})")
RE_TIMESLOT = re.compile(r"(\d{1,2})\s*時\s*~\s*(\d{1,2})\s*時")
RE_WATER = re.compile(r"水温:\s*(-?\d+(?:\.\d+)?)\s*℃")
RE_AIR = re.compile(r"気温:\s*(-?\d+(?:\.\d+)?)\s*℃")
RE_WIND = re.compile(r"風:\s*([東西南北]+)\s*(\d+(?:\.\d+)?)\s*m")
RE_CATCH = re.compile(r"^([^/\n]+?)/([^/\n]*?(cm|kg|g))/(\d+)\s*(?:尾|杯)/ポイント\s*([A-W~,\s]+)")
RE_NOANGLER = re.compile(r"釣り人なし|釣人なし")
RE_COMMENT = re.compile(r"^[★☆]|さん」|様$|さん$|様/|さん/")

def parse_tide_line(line):
    m = RE_TIDE_HEAD.match(line)
    if not m: return None
    tide = canon_tide(m.group(1)); rest = m.group(2)
    high, low = [], []
    if "干潮" in rest:
        before, after = rest.split("干潮", 1)
        before = before.replace("満潮", " ")
        high = [t for t in RE_TIME_OR_NA.findall(before) if ":" in t and "-" not in t]
        low = [t for t in RE_TIME_OR_NA.findall(after) if ":" in t and "-" not in t]
    else:
        rest2 = rest.replace("満潮", " ")
        high = [t for t in RE_TIME_OR_NA.findall(rest2) if ":" in t and "-" not in t]
    return tide, high, low

def expand_points(raw):
    raw = raw.strip().rstrip(","); out = []
    for token in raw.split(","):
        token = token.strip()
        if not token: continue
        mm = re.match(r"([A-W])\s*~\s*([A-W])", token)
        if mm:
            a, b = ord(mm.group(1)), ord(mm.group(2))
            if a <= b: out.extend(chr(c) for c in range(a, b+1))
        else:
            m2 = re.match(r"([A-W])", token)
            if m2: out.append(m2.group(1))
    seen, uniq = set(), []
    for p in out:
        if p not in seen: seen.add(p); uniq.append(p)
    return uniq

def parse_size(raw):
    nums = re.findall(r"(\d+(?:\.\d+)?)", raw)
    if not nums: return (None, None)
    vals = [float(n) for n in nums]
    return (min(vals), max(vals)) if len(vals) > 1 else (vals[0], vals[0])

def split_records(page_text):
    lines = page_text.splitlines(); idxs = []
    for i in range(len(lines)-1):
        if re.fullmatch(r"\s*\d{1,2}-\d{1,2}\s*", lines[i]) and RE_ISO_DATE.search(lines[i+1]):
            idxs.append(i)
    records = []
    for k, start in enumerate(idxs):
        end = idxs[k+1] if k+1 < len(idxs) else len(lines)
        records.append((lines[start].strip(), "\n".join(lines[start:end])))
    return records

def parse_record(heading, body):
    body = normalize(body); rows = []
    mdate = RE_ISO_DATE.search(body)
    date = f"{mdate.group(1)}-{mdate.group(2)}-{mdate.group(3)}" if mdate else None
    tide_type, high_tides, low_tides = None, [], []
    for line in body.splitlines():
        p = parse_tide_line(line)
        if p: tide_type, high_tides, low_tides = p; break
    lines = body.splitlines()
    slot_lines = [(i, RE_TIMESLOT.match(ln.strip())) for i, ln in enumerate(lines)]
    slot_lines = [(i, m) for i, m in slot_lines if m]
    day_water = RE_WATER.search(body); day_air = RE_AIR.search(body); day_wind = RE_WIND.search(body)
    for idx, (li, sm) in enumerate(slot_lines):
        seg_start = li
        seg_end = slot_lines[idx+1][0] if idx+1 < len(slot_lines) else len(lines)
        look_start = max(0, seg_start-1)
        segment = "\n".join(lines[look_start:seg_end])
        slot_label = f"{int(sm.group(1))}-{int(sm.group(2))}"
        water = RE_WATER.search(segment); air = RE_AIR.search(segment); wind = RE_WIND.search(segment)
        water_v = float(water.group(1)) if water else (float(day_water.group(1)) if day_water else None)
        air_v = float(air.group(1)) if air else (float(day_air.group(1)) if day_air else None)
        wind_dir = wind.group(1) if wind else (day_wind.group(1) if day_wind else None)
        wind_spd = float(wind.group(2)) if wind else (float(day_wind.group(2)) if day_wind else None)
        base = {"date":date,"tide_type":tide_type,"high_tides":"|".join(high_tides),
                "low_tides":"|".join(low_tides),"time_slot":slot_label,
                "water_temp":water_v,"air_temp":air_v,"wind_dir":wind_dir,"wind_speed":wind_spd}
        seg_body = "\n".join(lines[seg_start:seg_end]); got = False
        for line in seg_body.splitlines():
            line = line.strip().lstrip("*").strip()
            if RE_COMMENT.search(line): continue
            cm = RE_CATCH.match(line)
            if not cm: continue
            got = True
            species = cm.group(1).strip()
            smin, smax = parse_size(cm.group(2)); size_unit = cm.group(3)
            count = int(cm.group(4))
            points = expand_points(cm.group(5))
            for pt in points:
                rows.append({**base,"species":species,"size_min":smin,"size_max":smax,
                             "size_unit":size_unit,"count":count,"point":pt,
                             "n_points_shared":len(points)})
        if not got and RE_NOANGLER.search(seg_body):
            rows.append({**base,"species":None,"size_min":None,"size_max":None,
                         "size_unit":None,"count":0,"point":None,"n_points_shared":0})
    return rows

def parse_page(page_text):
    all_rows = []
    for h, b in split_records(page_text):
        all_rows.extend(parse_record(h, b))
    return all_rows

if __name__ == "__main__":
    import sys, csv
    with open(sys.argv[1], encoding="utf-8") as f:
        rows = parse_page(f.read())
    fields = ["date","tide_type","high_tides","low_tides","time_slot","water_temp","air_temp",
              "wind_dir","wind_speed","species","size_min","size_max","size_unit","count","point","n_points_shared"]
    w = csv.DictWriter(sys.stdout, fieldnames=fields); w.writeheader()
    for r in rows: w.writerow(r)
    print(f"\n# parsed {len(rows)} rows", file=sys.stderr)
