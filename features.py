# -*- coding: utf-8 -*-
"""
特徴量エンジニアリング: umizuri_catch.csv に、釣果予測に効く手がかりを追加する。

追加する主な特徴量:
  潮タイミング   … 各時間帯の中央時刻が「上げ/下げ潮の何時間目か」
                    「満潮/干潮まであと何時間か」
  月齢           … 新月からの経過日(大潮/小潮の連続指標)
  季節           … 月, 通日(1-366), sin/cosによる周期表現
  マヅメ         … 日の出/日の入りに時間帯が重なるか
  風             … 桟橋(北東沖出し)に対するオンショア成分(-1..+1)と風速の積
  ポイント属性   … zone(先端/通路/岸寄り), exposure(外洋への開け具合), corner

使い方:
    python3 features.py umizuri_catch.csv umizuri_features.csv

前提ライブラリ: pandas, astral, ephem
    python -m pip install astral ephem
"""
import sys, math, datetime
import pandas as pd

try:
    from astral import LocationInfo
    from astral.sun import sun
    import ephem
except ImportError:
    print("astral と ephem が必要です: python -m pip install astral ephem", file=sys.stderr)
    raise

# 海づり公園 概算位置
LOC = LocationInfo('Fukuoka', 'Japan', 'Asia/Tokyo', 33.56, 130.23)

# 桟橋先端の外洋向き方位(北東=45度)
OFFSHORE_BEARING = 45.0
WIND_DIRS = {'北':0,'北北東':22.5,'北東':45,'東北東':67.5,'東':90,'東南東':112.5,
    '南東':135,'南南東':157.5,'南':180,'南南西':202.5,'南西':225,'西南西':247.5,
    '西':270,'西北西':292.5,'北西':315,'北北西':337.5}

POINT_ATTR = {
    'A':('tip',0.9,1),'B':('tip',0.9,0),'C':('tip',0.9,0),'D':('tip',1.0,0),
    'E':('tip',0.9,0),'F':('tip',0.9,1),'G':('tip',0.8,1),'H':('tip',0.8,0),
    'I':('tip',0.8,0),'J':('tip',0.85,0),'K':('tip',0.8,0),'L':('tip',0.8,1),
    'M':('corridor',0.6,0),'Q':('corridor',0.6,0),'N':('corridor',0.5,0),'R':('corridor',0.5,0),
    'O':('inner',0.3,1),'P':('inner',0.2,1),'S':('inner',0.35,0),'T':('inner',0.35,0),
    'U':('inner',0.3,1),'V':('inner',0.3,0),'W':('inner',0.25,1)}

SLOT_CENTER = {'6-9':7.5,'9-12':10.5,'12-15':13.5,'15-18':16.5,'18-20':19.0}

def to_hours(hhmm):
    try:
        h,m = hhmm.split(':'); return int(h)+int(m)/60
    except Exception:
        return None

def nearest_tide_timing(center_h, tide_str):
    """時間帯中央時刻に対し、最も近い潮時刻までの時間(絶対値・符号付き)を返す。"""
    if not isinstance(tide_str,str) or not tide_str:
        return None
    times = [to_hours(t) for t in tide_str.split('|') if to_hours(t) is not None]
    if not times: return None
    diffs = [center_h - t for t in times]      # +:潮時刻を過ぎた  -:これから
    return min(diffs, key=abs)

# 天文計算はキャッシュ(同一日付の再計算を避ける)
_moon_cache, _sun_cache = {}, {}
def moon_age(d):
    if d not in _moon_cache:
        pn = ephem.previous_new_moon(d)
        _moon_cache[d] = float(ephem.Date(d) - pn)
    return _moon_cache[d]
def sun_times(d):
    if d not in _sun_cache:
        s = sun(LOC.observer, date=d, tzinfo='Asia/Tokyo')
        _sun_cache[d] = (s['sunrise'].hour+s['sunrise'].minute/60,
                         s['sunset'].hour+s['sunset'].minute/60)
    return _sun_cache[d]

def main(inp, out):
    df = pd.read_csv(inp)
    df['dt'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['dt']).copy()

    df['center_h'] = df['time_slot'].map(SLOT_CENTER)
    df['month'] = df['dt'].dt.month
    df['doy'] = df['dt'].dt.dayofyear
    df['doy_sin'] = (2*math.pi*df['doy']/366).apply(math.sin)
    df['doy_cos'] = (2*math.pi*df['doy']/366).apply(math.cos)

    # 潮タイミング
    df['to_high_h'] = df.apply(lambda r: nearest_tide_timing(r['center_h'], r['high_tides']), axis=1)
    df['to_low_h']  = df.apply(lambda r: nearest_tide_timing(r['center_h'], r['low_tides']), axis=1)

    # 天文(日付単位)
    uniq = {d.date() for d in df['dt']}
    ma = {d: moon_age(d) for d in uniq}
    st = {d: sun_times(d) for d in uniq}
    df['moon_age'] = df['dt'].dt.date.map(ma)
    df['sunrise_h'] = df['dt'].dt.date.map(lambda d: st[d][0])
    df['sunset_h']  = df['dt'].dt.date.map(lambda d: st[d][1])
    # マヅメ近さ: 日の出/日の入りに時間帯中央がどれだけ近いか(1に近いほど重なる)
    df['near_sunrise'] = (1 - (df['center_h']-df['sunrise_h']).abs()/3).clip(lower=0)
    df['near_sunset']  = (1 - (df['center_h']-df['sunset_h']).abs()/3).clip(lower=0)

    # 風
    df['wind_deg'] = df['wind_dir'].map(WIND_DIRS)
    df['onshore'] = df['wind_deg'].apply(
        lambda deg: math.cos(math.radians(deg-OFFSHORE_BEARING)) if pd.notna(deg) else None)
    df['onshore_strength'] = df['onshore'] * df['wind_speed']

    # ポイント属性
    df['zone'] = df['point'].map(lambda p: POINT_ATTR.get(p,('unknown',None,None))[0] if pd.notna(p) else None)
    df['exposure'] = df['point'].map(lambda p: POINT_ATTR.get(p,(None,None,None))[1] if pd.notna(p) else None)
    df['corner'] = df['point'].map(lambda p: POINT_ATTR.get(p,(None,None,None))[2] if pd.notna(p) else None)

    df = df.drop(columns=['dt'])
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"wrote {len(df):,} rows, {df.shape[1]} columns -> {out}", file=sys.stderr)
    print("added features:", ['center_h','month','doy_sin','doy_cos','to_high_h','to_low_h',
          'moon_age','near_sunrise','near_sunset','onshore','onshore_strength',
          'zone','exposure','corner'], file=sys.stderr)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv)>1 else "umizuri_catch.csv"
    out = sys.argv[2] if len(sys.argv)>2 else "umizuri_features.csv"
    main(inp, out)
