# -*- coding: utf-8 -*-
"""
明日の釣果予報(中核・第一段階): 明日の環境条件を与えると、
全ポイント×全時間帯について対象魚の「いる確率」を高い順に一覧表示する。

第一段階では明日の条件を設定ファイル(tomorrow.txt)から読む。
第二段階でこの条件を気象API・潮汐計算・サイトの直近水温から自動生成する。

使い方:
    python3 predict.py umizuri_clean.csv クロ tomorrow.txt

tomorrow.txt の書式(1行1項目, #はコメント):
    date=2026-07-03
    tide_type=中潮
    high_tides=11:30|23:50      # 満潮(複数は|区切り)
    low_tides=05:30|18:10       # 干潮
    air_temp=27                 # 明日の気温(時間帯共通の代表値でも可)
    wind_dir=北東
    wind_speed=3.0
    weather=晴れ                # 晴れ/快晴/曇り/雨/大雨
    recent_water=23.0,23.4,23.8 # 直近数日の実測水温(古い→新しい)
"""
import sys, math, datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

# features.py と同じ定義を再掲(整合性のため)
try:
    from astral import LocationInfo
    from astral.sun import sun
    import ephem
except ImportError:
    print("astral と ephem が必要: python -m pip install astral ephem", file=sys.stderr); raise

LOC = LocationInfo('Fukuoka','Japan','Asia/Tokyo',33.56,130.23)
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
FEATURES_NUM = ["water_temp","air_temp","wind_speed","center_h","month",
                "doy_sin","doy_cos","to_high_h","to_low_h","moon_age",
                "near_sunrise","near_sunset","onshore","onshore_strength","exposure","corner"]
FEATURES_CAT = ["tide_type","time_slot","point","zone","wind_dir"]

def estimate_water_temp(recent, weather):
    r = [t for t in recent if t is not None]
    if not r: return None
    base = r[-1]
    diffs = [r[i+1]-r[i] for i in range(max(0,len(r)-3), len(r)-1)]
    daily = float(np.mean(diffs)) if diffs else 0.0
    adj = {'晴れ':+0.4,'快晴':+0.5,'曇り':0.0,'雨':-0.4,'大雨':-0.7}
    return round(base + daily + adj.get(weather,0.0), 1)

def to_hours(hhmm):
    try: h,m = hhmm.split(':'); return int(h)+int(m)/60
    except: return None

def nearest_tide_timing(center_h, tide_str):
    if not tide_str: return None
    times=[to_hours(t) for t in tide_str.split('|') if to_hours(t) is not None]
    if not times: return None
    return min([center_h-t for t in times], key=abs)

def read_conditions(path):
    cond={}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            if '=' in line:
                k,v=line.split('=',1); cond[k.strip()]=v.split('#')[0].strip()
    return cond

def main(clean_csv, target, cond_path):
    cond = read_conditions(cond_path)
    date = cond['date']
    d = datetime.date.fromisoformat(date)
    weather = cond.get('weather','曇り')
    recent = [float(x) for x in cond.get('recent_water','').split(',') if x.strip()]
    water = estimate_water_temp(recent, weather)
    if water is None:
        print("recent_water(直近水温)が必要です", file=sys.stderr); return

    # 天文(明日)
    pn=ephem.previous_new_moon(d); moon_age=float(ephem.Date(d)-pn)
    s=sun(LOC.observer,date=d,tzinfo='Asia/Tokyo')
    sunrise=s['sunrise'].hour+s['sunrise'].minute/60
    sunset=s['sunset'].hour+s['sunset'].minute/60
    doy=d.timetuple().tm_yday

    wind_dir=cond.get('wind_dir'); wind_speed=float(cond.get('wind_speed',0))
    wind_deg=WIND_DIRS.get(wind_dir)
    onshore=math.cos(math.radians(wind_deg-OFFSHORE_BEARING)) if wind_deg is not None else 0.0

    # 学習に登場したポイント一覧をデータから取得(存在するものだけ予報)
    df=pd.read_csv(clean_csv)
    points=sorted([p for p in df['point'].dropna().unique() if p in POINT_ATTR])

    rows=[]
    for slot,center in SLOT_CENTER.items():
        for p in points:
            zone,exposure,corner=POINT_ATTR[p]
            rows.append(dict(
                water_temp=water, air_temp=float(cond.get('air_temp',water)),
                wind_speed=wind_speed, center_h=center, month=d.month,
                doy_sin=math.sin(2*math.pi*doy/366), doy_cos=math.cos(2*math.pi*doy/366),
                to_high_h=nearest_tide_timing(center,cond.get('high_tides','')),
                to_low_h=nearest_tide_timing(center,cond.get('low_tides','')),
                moon_age=moon_age,
                near_sunrise=max(0,1-abs(center-sunrise)/3),
                near_sunset=max(0,1-abs(center-sunset)/3),
                onshore=onshore, onshore_strength=onshore*wind_speed,
                exposure=exposure, corner=corner,
                tide_type=cond.get('tide_type'), time_slot=slot, point=p,
                zone=zone, wind_dir=wind_dir))
    X=pd.DataFrame(rows)
    for c in FEATURES_CAT: X[c]=X[c].astype('category')

    import hashlib
    _tag=hashlib.md5(target.encode('utf-8')).hexdigest()[:8]
    model=lgb.Booster(model_file=f"model_{_tag}.txt")
    X['prob']=model.predict(X[FEATURES_NUM+FEATURES_CAT])

    # 表示
    print(f"=== {date} {target} 釣果予報 ===")
    print(f"推定水温 {water}℃ (直近{recent} → {weather}補正)  "
          f"気温{cond.get('air_temp')}  風 {wind_dir}{wind_speed}m  {cond.get('tide_type')}")
    print(f"月齢{moon_age:.1f}  日の出{sunrise:.1f}時 日の入{sunset:.1f}時")
    print()
    out=X[['time_slot','point','prob']].sort_values('prob',ascending=False)
    out['prob']=(out['prob']*100).round(1)
    print("釣れる確率が高い順(上位25):")
    print(out.head(25).to_string(index=False))
    out.to_csv(f"forecast_{target.replace('*','_star')}_{date}.csv", index=False, encoding='utf-8-sig')
    print(f"\n全結果を forecast_{target.replace('*','_star')}_{date}.csv に保存")

if __name__ == "__main__":
    clean = sys.argv[1] if len(sys.argv)>1 else "umizuri_clean.csv"
    target = sys.argv[2] if len(sys.argv)>2 else "クロ"
    cond = sys.argv[3] if len(sys.argv)>3 else "tomorrow.txt"
    main(clean, target, cond)
