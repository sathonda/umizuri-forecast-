# -*- coding: utf-8 -*-
"""
第二段階: 明日の環境条件を自動生成して tomorrow.txt を書き出す。

取得元:
  潮回り      … 月齢から分類(ephem, 確実)
  満干時刻    … 収集済みデータ内の「同じ潮回り・近い月齢」の日から推定
                (港湾地形依存のため天文計算では高精度化できないための代替)
  気温/風/天気 … Open-Meteo 気象API(無料・キー不要)から時間帯別に取得
  直近水温    … 海づり公園サイトの最近の釣果ページから数日分を取得

使い方:
    python3 auto_conditions.py umizuri_clean.csv            # 明日について生成
    python3 auto_conditions.py umizuri_clean.csv 2026-07-05 # 指定日

生成後、必要なら tomorrow.txt を手で微修正してから predict.py を実行できる。
"""
import sys, math, datetime, json, urllib.request, re
import pandas as pd

try:
    import ephem
except ImportError:
    print("ephem が必要: python -m pip install ephem", file=sys.stderr); raise

LAT, LON = 33.56, 130.23
SLOT_CENTER = {'6-9':7.5,'9-12':10.5,'12-15':13.5,'15-18':16.5,'18-20':19.0}
DIRS16=['北','北北東','北東','東北東','東','東南東','南東','南南東','南','南南西','南西','西南西','西','西北西','北西','北北西']
WMO={0:'快晴',1:'晴れ',2:'曇り',3:'曇り',45:'曇り',48:'曇り',51:'雨',53:'雨',55:'雨',
     56:'雨',57:'雨',61:'雨',63:'雨',65:'大雨',66:'雨',67:'大雨',71:'雨',73:'雨',75:'大雨',
     77:'雨',80:'雨',81:'雨',82:'大雨',85:'雨',86:'大雨',95:'大雨',96:'大雨',99:'大雨'}

def deg_to_jp(deg):
    return DIRS16[round(deg/22.5)%16]

def tide_type_from_age(a):
    a=a%29.53
    dist=min(abs(a-0),abs(a-14.77),abs(a-29.53))
    return '大潮' if dist<=1.5 else '中潮' if dist<=3.5 else '小潮' if dist<=5.5 else '長潮' if dist<=6.5 else '若潮'

def moon_age(d):
    return float(ephem.Date(d)-ephem.previous_new_moon(d))

def estimate_tides(df, tide_type, m_age):
    """収集データから、同じ潮回りで月齢が近い日の満干時刻の代表値を推定。"""
    sub=df[df['tide_type']==tide_type].dropna(subset=['high_tides'])
    if 'moon_age' in sub.columns:
        sub=sub.assign(agediff=(sub['moon_age']-m_age).abs()).sort_values('agediff')
    def rep_times(col):
        # 最も月齢が近い日の値をそのまま採用(1件)
        for v in sub[col].dropna():
            if isinstance(v,str) and v.strip(): return v
        return ''
    # 日単位で1件取ればよい
    hi = sub['high_tides'].dropna().iloc[0] if len(sub) else ''
    lo = sub['low_tides'].dropna().iloc[0] if 'low_tides' in sub and len(sub['low_tides'].dropna()) else ''
    return hi, lo

# Open-Meteo の無料予報は概ね16日先まで。これを超える日は実際の気象予報が
# 存在しないため、過去データの季節平均(気候値)で代替する。
FORECAST_HORIZON_DAYS = 16

def fetch_weather(date):
    """Open-Meteoから指定日の時間別 気温/風速/風向/天気コードを取得。
    変数名の綴り違い(weather_code/weathercode 等)に備えて複数パターンを試し、
    400等では本文の理由を含めて例外にする。"""
    import urllib.error
    variants = [
        ("temperature_2m,wind_speed_10m,wind_direction_10m,weather_code", "wind_speed_unit"),
        ("temperature_2m,windspeed_10m,winddirection_10m,weathercode",   "windspeed_unit"),
    ]
    last_err = None
    for hourly, spd_unit in variants:
        url=(f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
             f"&hourly={hourly}&{spd_unit}=ms&timezone=Asia/Tokyo"
             f"&start_date={date}&end_date={date}")
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                d=json.load(r)
            h=d['hourly']
            # 綴り差を吸収して共通キーへ揃える
            return {
                'time': h.get('time', []),
                'temperature_2m': h.get('temperature_2m'),
                'wind_speed_10m': h.get('wind_speed_10m', h.get('windspeed_10m')),
                'wind_direction_10m': h.get('wind_direction_10m', h.get('winddirection_10m')),
                'weather_code': h.get('weather_code', h.get('weathercode')),
            }
        except urllib.error.HTTPError as e:
            body=e.read().decode('utf-8','ignore'); reason=body
            try: reason=json.loads(body).get('reason', body)
            except Exception: pass
            last_err=RuntimeError(f"HTTP {e.code}: {reason[:200]}")
        except Exception as e:
            last_err=e
    raise last_err if last_err else RuntimeError("unknown error")

def climatology_weather(df, d, window_days=10):
    """予報が無い先の日付向け: 過去データから同じ時期(±window_days、複数年)の
    平均的な気温・風を気候値として返す。天気は不明として曇り扱い。"""
    if 'air_temp' not in df.columns:
        return None
    target_doy = d.timetuple().tm_yday
    dd = df.copy()
    dd['dt'] = pd.to_datetime(dd['date'], errors='coerce')
    dd = dd.dropna(subset=['dt'])
    dd['doy'] = dd['dt'].dt.dayofyear
    # 年末年始をまたぐ距離も考慮した通日差
    diff = (dd['doy'] - target_doy).abs()
    diff = diff.where(diff <= 183, 366 - diff)
    sub = dd[diff <= window_days]
    if sub.empty:
        return None
    air = sub['air_temp'].dropna()
    spd = sub['wind_speed'].dropna()
    # 風向は最頻値(その時期に多い向き)
    wdir = sub['wind_dir'].dropna()
    return dict(
        air_temp = round(float(air.mean()), 1) if len(air) else None,
        wind_speed = round(float(spd.mean()), 1) if len(spd) else None,
        wind_dir = wdir.mode().iloc[0] if len(wdir) else None,
        n_days = int(sub['date'].nunique()))

def hour_index(times, target_hour):
    for i,t in enumerate(times):
        if int(t[11:13])==target_hour: return i
    return None

def fetch_recent_water():
    """海づり公園サイトの最新ページから直近の水温を数日分拾う。"""
    from bs4 import BeautifulSoup
    url="https://umizuri.com/pages/30/p184=1"
    req=urllib.request.Request(url, headers={'User-Agent':'umizuri-forecast/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        html=r.read().decode('utf-8','ignore')
    soup=BeautifulSoup(html,'html.parser')
    text=(soup.find('article') or soup).get_text('\n',strip=True)
    text=text.replace('：',':')
    temps=[float(x) for x in re.findall(r'水温:\s*(-?\d+(?:\.\d+)?)\s*℃', text)]
    # 直近数日ぶんの代表(先頭=最新側)。古い→新しい順に3件返す。
    recent=list(reversed(temps[:6]))[-3:] if temps else []
    return recent

def main(clean_csv, date=None):
    if date is None:
        date=(datetime.date.today()+datetime.timedelta(days=1)).isoformat()
    d=datetime.date.fromisoformat(date)
    df=pd.read_csv(clean_csv)

    m_age=moon_age(d)
    tide=tide_type_from_age(m_age)
    hi,lo=estimate_tides(df,tide,m_age)

    # 何日先か
    days_ahead = (d - datetime.date.today()).days

    # 気象(代表として9-12時=昼前の値を全体の代表に。時間帯別は predict 側で気温共通利用)
    weather_note=""
    air=None; wdir=None; wspd=None; weather='曇り'
    source="予報"   # 予報 / 気候値 / 手入力
    if days_ahead > FORECAST_HORIZON_DAYS:
        # 予報の範囲外 → 過去データの季節平均(気候値)で代替
        clim = climatology_weather(df, d)
        if clim and clim['air_temp'] is not None:
            air=clim['air_temp']; wdir=clim['wind_dir']; wspd=clim['wind_speed']
            weather='曇り'; source="気候値"
            weather_note=(f"# 【気候値】{date} は本日から{days_ahead}日先で気象予報の範囲外"
                          f"({FORECAST_HORIZON_DAYS}日先まで)です。\n"
                          f"# 過去{clim['n_days']}日分の同時期データの平均で代用しています"
                          f"(実際の予報ではありません)。\n"
                          f"# 日が近づいたら再実行すると実際の予報に置き換わります。\n")
        else:
            weather_note=(f"# {date} は予報範囲外({FORECAST_HORIZON_DAYS}日先まで)で、"
                          f"気候値も作れませんでした。air_temp/wind/weather を手で記入してください\n")
            source="手入力"
    else:
        try:
            h=fetch_weather(date)
            idx=hour_index(h['time'],12) or 12
            air=round(h['temperature_2m'][idx],1)
            wdir=deg_to_jp(h['wind_direction_10m'][idx])
            wspd=round(h['wind_speed_10m'][idx],1)
            weather=WMO.get(h['weather_code'][idx],'曇り')
        except Exception as e:
            weather_note=f"# 気象API取得に失敗({e}). 手で air_temp/wind/weather を記入してください\n"
            source="手入力"

    # 直近水温。予報範囲内なら実測を使う。範囲外(先の計画)や取得失敗時は、
    # 過去データの同時期平均(気候値)を water として使い、パイプラインを止めない。
    water_note=""
    recent=[]
    def water_climatology():
        c = climatology_weather(df, d)
        # climatology_weather は air/wind を返すが、water は別途 water_temp 列から算出
        if 'water_temp' not in df.columns: return None
        dd=df.copy(); dd['dt']=pd.to_datetime(dd['date'],errors='coerce')
        dd=dd.dropna(subset=['dt']); dd['doy']=dd['dt'].dt.dayofyear
        target=d.timetuple().tm_yday
        diff=(dd['doy']-target).abs(); diff=diff.where(diff<=183,366-diff)
        w=dd[diff<=10]['water_temp'].dropna()
        return round(float(w.mean()),1) if len(w) else None

    if days_ahead > FORECAST_HORIZON_DAYS:
        wclim = water_climatology()
        if wclim is not None:
            recent=[wclim]
            water_note += (f"# 水温は過去の同時期平均(気候値){wclim}℃ を使用しています"
                           f"(先の予定のため実測ではありません)。\n")
        else:
            water_note += "# 水温の気候値が作れませんでした。recent_water を手で記入してください\n"
    else:
        try:
            recent=fetch_recent_water()
            if not recent: raise ValueError('水温が見つからない')
        except Exception as e:
            wclim = water_climatology()
            if wclim is not None:
                recent=[wclim]
                water_note += (f"# 直近水温の取得に失敗({e})。過去の同時期平均{wclim}℃ で代用。\n")
            else:
                water_note += f"# 直近水温の取得に失敗({e}). recent_water を手で記入してください\n"

    lines=[f"# 自動生成 {datetime.datetime.now():%Y-%m-%d %H:%M}  対象日 {date}",
           weather_note, water_note,
           f"date={date}",
           f"tide_type={tide}",
           f"high_tides={hi}",
           f"low_tides={lo}",
           f"air_temp={air if air is not None else ''}",
           f"wind_dir={wdir if wdir else ''}",
           f"wind_speed={wspd if wspd is not None else ''}",
           f"weather={weather}",
           f"recent_water={','.join(str(x) for x in recent)}"]
    out="\n".join(l for l in lines if l!="")+"\n"
    with open("tomorrow.txt","w",encoding="utf-8") as f:
        f.write(out)
    print(out)
    # 計画の見通しを分かりやすく伝える
    print(f"対象日 {date} は本日から {days_ahead} 日先です。気象の出どころ: {source}")
    if source=="予報":
        remain = FORECAST_HORIZON_DAYS - days_ahead
        print(f"  実際の気象予報を使用。あと約{remain}日先(最大{FORECAST_HORIZON_DAYS}日先)まで"
              f"同様に予報できます。")
    elif source=="気候値":
        print("  予報範囲外のため季節平均で代用しています。潮回りは正確ですが、")
        print("  気温・風は目安です。日が近づいて再実行すると実際の予報に変わります。")
    else:
        print("  気象値が取得できませんでした。tomorrow.txt を手で記入してください。")
    print("→ tomorrow.txt を生成しました。内容を確認し、必要なら手修正して")
    print("   python predict_map.py umizuri_clean.csv tomorrow.txt")

if __name__ == "__main__":
    clean=sys.argv[1] if len(sys.argv)>1 else "umizuri_clean.csv"
    date=sys.argv[2] if len(sys.argv)>2 else None
    main(clean, date)
