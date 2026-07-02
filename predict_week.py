# -*- coding: utf-8 -*-
"""
複数日いっぺんに予報して、日付リボン付きの色分け地図HTMLを1枚作る。

predict_map.py は「1日分」でしたが、これは本日から指定日数ぶん(既定16日=
気象予報の届く範囲)をまとめて計算し、1つのHTMLに埋め込みます。ブラウザ上で
【日付】【魚種】【時間帯】の3つのボタンで切り替えて閲覧できます。

気象は Open-Meteo から範囲まとめて1回で取得(16日でもAPI呼び出しは1回)。
潮回り・満干は各日ごとに天文計算。水温は直近実測を起点に日ごとへ延ばします。

使い方:
    python predict_week.py umizuri_clean.csv              # 本日から16日分
    python predict_week.py umizuri_clean.csv --days 10    # 日数を変える
    python predict_week.py umizuri_clean.csv --start 2026-07-10 --days 7
                                                          # 起点日を指定

前提: train_all.py で各魚種モデル(model_*.txt)を学習済みであること。
出力: forecast_week_<起点日>.html (ダブルクリックでブラウザ表示)
"""
import sys, math, datetime, json, os, argparse, urllib.request
import numpy as np
import pandas as pd
import lightgbm as lgb

import auto_conditions as ac
import predict_map as pm

LAT, LON = ac.LAT, ac.LON
SLOTS = pm.SLOTS
COORDS = pm.COORDS
TARGETS = pm.TARGETS
FEATURES_NUM = pm.FEATURES_NUM
FEATURES_CAT = pm.FEATURES_CAT
FORECAST_HORIZON_DAYS = getattr(ac, "FORECAST_HORIZON_DAYS", 16)

# Open-Meteo は時期により変数名の綴りが違うこと(weather_code / weathercode など)が
# あり、綴り違いは 400 Bad Request の主因になる。複数の綴りを順に試す。
_HOURLY_VARIANTS = [
    ("temperature_2m,wind_speed_10m,wind_direction_10m,weather_code", "wind_speed_unit"),
    ("temperature_2m,windspeed_10m,winddirection_10m,weathercode",   "windspeed_unit"),
]

def _http_get_json(url):
    """URLを取得しJSONを返す。400等では本文の理由(reason)を含めて例外にする。"""
    import urllib.error
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        reason = body
        try:
            j = json.loads(body)
            reason = j.get("reason", body)
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {reason[:200]}")

def fetch_weather_range(start, end):
    """開始〜終了日の時間別気象を1回のAPI呼び出しで取得。{日付: 時間別dict} を返す。

    Open-Meteo の標準 /v1/forecast は既定7日で、16日先まで得るには forecast_days=
    を指定する必要がある(start_date/end_date で長い範囲を指定すると 400 になる)。
    そこで、起点が本日なら forecast_days= を使い、起点が先の日付なら start_date/
    end_date を使う。変数名の綴り違いにも備えて複数パターンを試す。"""
    start_d=datetime.date.fromisoformat(start)
    end_d=datetime.date.fromisoformat(end)
    today=datetime.date.today()
    span_days=(end_d-today).days+1   # 本日から終了日までに必要な日数

    last_err=None
    for hourly, spd_unit in _HOURLY_VARIANTS:
        # まず forecast_days 方式(本日起点、~16日先まで対応)
        attempts=[]
        if start_d==today and 1<=span_days<=16:
            attempts.append(
                f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
                f"&hourly={hourly}&{spd_unit}=ms&timezone=Asia/Tokyo&forecast_days={span_days}")
        # 次に start_date/end_date 方式(先の起点日など)
        attempts.append(
            f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
            f"&hourly={hourly}&{spd_unit}=ms&timezone=Asia/Tokyo&start_date={start}&end_date={end}")
        for url in attempts:
            try:
                d=_http_get_json(url)
                h=_normalize_hourly(d["hourly"])
                by_date={}
                for i,t in enumerate(h["time"]):
                    by_date.setdefault(t[:10], []).append((t, i))
                return h, by_date
            except Exception as e:
                last_err=e
                continue
    raise RuntimeError(str(last_err) if last_err else "unknown error")

def _normalize_hourly(h):
    """新旧の綴り(weather_code/weathercode 等)を共通キーに揃える。"""
    def pick(*names):
        for n in names:
            if n in h: return h[n]
        return None
    return {
        "time": h.get("time", []),
        "temperature_2m": pick("temperature_2m"),
        "wind_speed_10m": pick("wind_speed_10m","windspeed_10m"),
        "wind_direction_10m": pick("wind_direction_10m","winddirection_10m"),
        "weather_code": pick("weather_code","weathercode"),
    }

def day_weather(h, by_date, date_str):
    """指定日の代表気象(12時の値)を返す。無ければ None。"""
    if date_str not in by_date: return None
    # 12時に最も近い時刻を選ぶ
    best=None
    for t,i in by_date[date_str]:
        hour=int(t[11:13])
        if best is None or abs(hour-12)<abs(best[0]-12):
            best=(hour,i)
    i=best[1]
    return dict(
        air_temp=round(h["temperature_2m"][i],1) if h["temperature_2m"][i] is not None else None,
        wind_dir=ac.deg_to_jp(h["wind_direction_10m"][i]) if h["wind_direction_10m"][i] is not None else None,
        wind_speed=round(h["wind_speed_10m"][i],1) if h["wind_speed_10m"][i] is not None else None,
        weather=ac.WMO.get(h["weather_code"][i],"曇り"))

def build_conditions_for_day(df, d, wx, base_water):
    """1日分の cond 辞書(predict_map が読む形)を組み立てる。
    wx=その日の気象(None可)、base_water=起点の直近水温リスト。"""
    m_age=ac.moon_age(d)
    tide=ac.tide_type_from_age(m_age)
    hi,lo=ac.estimate_tides(df,tide,m_age)
    days_ahead=(d-datetime.date.today()).days

    if wx and wx["air_temp"] is not None:
        air=wx["air_temp"]; wdir=wx["wind_dir"]; wspd=wx["wind_speed"]; weather=wx["weather"]
        wsrc="予報"
    else:
        clim=ac.climatology_weather(df,d)
        if clim and clim["air_temp"] is not None:
            air=clim["air_temp"]; wdir=clim["wind_dir"]; wspd=clim["wind_speed"]; weather="曇り"
            wsrc="気候値"
        else:
            air=wdir=wspd=None; weather="曇り"; wsrc="不明"

    # 水温: 起点の直近実測があればそれを、無ければ季節平均を使う
    if base_water:
        recent=base_water
    else:
        wc=_water_climatology(df,d)
        recent=[wc] if wc is not None else []

    cond=dict(date=d.isoformat(), tide_type=tide,
              high_tides=hi, low_tides=lo,
              air_temp="" if air is None else air,
              wind_dir="" if not wdir else wdir,
              wind_speed="" if wspd is None else wspd,
              weather=weather,
              recent_water=",".join(str(x) for x in recent))
    return cond, wsrc, dict(moon_age=round(m_age,1))

def _water_climatology(df, d, window=10):
    if "water_temp" not in df.columns: return None
    dd=df.copy(); dd["dt"]=pd.to_datetime(dd["date"],errors="coerce")
    dd=dd.dropna(subset=["dt"]); dd["doy"]=dd["dt"].dt.dayofyear
    target=d.timetuple().tm_yday
    diff=(dd["doy"]-target).abs(); diff=diff.where(diff<=183,366-diff)
    w=dd[diff<=window]["water_temp"].dropna()
    return round(float(w.mean()),1) if len(w) else None

def predict_one_day(cond, df, points, models):
    """1日分の probs[species][slot][point] と 表示用メタを返す。"""
    recent=[float(x) for x in cond.get("recent_water","").split(",") if x.strip()]
    water=pm.estimate_water_temp(recent, cond.get("weather","曇り"))
    if water is None:
        return None, None
    X,astro=pm.build_features(cond,water,points)
    # 潮時刻が空の日など、to_high_h/to_low_h が全て None になると object 型になり
    # LightGBM が受け付けない。数値列は明示的に float 化(None→NaN)する。
    for c in FEATURES_NUM:
        X[c]=pd.to_numeric(X[c], errors="coerce")
    for c in FEATURES_CAT: X[c]=X[c].astype("category")
    probs={}
    for t,model in models.items():
        X["_p"]=model.predict(X[FEATURES_NUM+FEATURES_CAT])
        dd={}
        for slot in SLOTS:
            sub=X[X["time_slot"]==slot]
            dd[slot]={r["point"]:round(float(r["_p"])*100,1) for _,r in sub.iterrows()}
        probs[t]=dd
    meta=dict(water=water, air=cond.get("air_temp"), wind_dir=cond.get("wind_dir"),
              wind_speed=cond.get("wind_speed"), tide=cond.get("tide_type"),
              weather=cond.get("weather"),
              moon_age=astro["moon_age"], sunrise=astro["sunrise"], sunset=astro["sunset"])
    return probs, meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("clean_csv", nargs="?", default="umizuri_clean.csv")
    ap.add_argument("--days", type=int, default=FORECAST_HORIZON_DAYS,
                    help=f"何日分予報するか(既定{FORECAST_HORIZON_DAYS})")
    ap.add_argument("--start", default=None, help="起点日 YYYY-MM-DD(既定=本日)")
    args=ap.parse_args()

    df=pd.read_csv(args.clean_csv)
    points=sorted([p for p in df["point"].dropna().unique() if p in pm.POINT_ATTR])

    # モデル読み込み(1回だけ)
    models={}
    for t in TARGETS:
        mf=pm.model_filename(t)
        if os.path.exists(mf):
            models[t]=lgb.Booster(model_file=mf)
        else:
            print(f"  モデル未学習でスキップ: {t} ({mf})", file=sys.stderr)
    if not models:
        print("学習済みモデルがありません。先に train_all.py を実行してください。", file=sys.stderr)
        return
    trained=list(models.keys())

    start=datetime.date.fromisoformat(args.start) if args.start else datetime.date.today()
    dates=[start+datetime.timedelta(days=i) for i in range(args.days)]

    # 気象をまとめて1回取得(範囲内のぶん)。失敗したら日別取得に切り替え、
    # それも駄目なら季節平均(気候値)へ自動フォールバック。
    h=by_date=None
    per_day={}   # date_iso -> wx dict (日別取得できた分)
    try:
        h,by_date=fetch_weather_range(dates[0].isoformat(), dates[-1].isoformat())
        got=len([d for d in dates if d.isoformat() in by_date])
        print(f"気象を一括取得: {got}/{len(dates)}日分", file=sys.stderr)
    except Exception as e:
        print(f"気象の一括取得に失敗({e})。", file=sys.stderr)
        print("→ 日別取得に切り替えて再試行します...", file=sys.stderr)
        today=datetime.date.today()
        for d in dates:
            # 予報範囲(本日+16日)内の日だけAPIを叩く。範囲外は気候値に任せる。
            if not (0 <= (d-today).days <= FORECAST_HORIZON_DAYS):
                continue
            try:
                hh=ac.fetch_weather(d.isoformat())
                idx=ac.hour_index(hh["time"],12)
                if idx is None: idx=min(12,len(hh["time"])-1)
                per_day[d.isoformat()]=dict(
                    air_temp=round(hh["temperature_2m"][idx],1),
                    wind_dir=ac.deg_to_jp(hh["wind_direction_10m"][idx]),
                    wind_speed=round(hh["wind_speed_10m"][idx],1),
                    weather=ac.WMO.get(hh["weather_code"][idx],"曇り"))
            except Exception as e2:
                # 日別も失敗(最初の1件で理由を表示し、以降は静かに気候値へ)
                if not per_day:
                    print(f"   日別取得も失敗({e2})。取得できない日は季節平均(気候値)で計算します。",
                          file=sys.stderr)
        if per_day:
            print(f"   日別取得: {len(per_day)}日分を実際の予報で取得", file=sys.stderr)

    # 起点の直近水温(サイトから)。取れなければ各日で季節平均。
    try:
        base_water=ac.fetch_recent_water()
    except Exception:
        base_water=[]

    all_data={}   # date -> {species: {slot: {point: prob}}}
    day_meta={}   # date -> 表示メタ
    src_count={"予報":0,"気候値":0,"不明":0}
    for d in dates:
        diso=d.isoformat()
        # 気象の出どころ: 一括 → 日別 → (無ければ)気候値
        wx=None
        if by_date and diso in by_date:
            wx=day_weather(h,by_date,diso)
        elif diso in per_day:
            wx=per_day[diso]
        cond,wsrc,_=build_conditions_for_day(df,d,wx,base_water)
        src_count[wsrc]=src_count.get(wsrc,0)+1
        probs,meta=predict_one_day(cond,df,points,models)
        if probs is None:
            print(f"  {d} は水温不明でスキップ", file=sys.stderr); continue
        meta["wsrc"]=wsrc
        all_data[d.isoformat()]=probs
        day_meta[d.isoformat()]=meta
        print(f"  {d} ({wsrc}) 完了", file=sys.stderr)

    if not all_data:
        print("予報できた日がありません。", file=sys.stderr); return

    html=render_week_html(all_data, day_meta, trained, points, start)
    out=f"forecast_week_{start.isoformat()}.html"
    with open(out,"w",encoding="utf-8") as f: f.write(html)
    print(f"\n生成: {out}")
    print(f"  {len(all_data)}日 × 魚種{len(trained)} × 時間帯{len(SLOTS)}")
    print(f"  気象内訳: 予報{src_count.get('予報',0)}日 / 気候値{src_count.get('気候値',0)}日")
    print("ダブルクリックでブラウザ表示。日付・魚種・時間帯のボタンで切り替え。")

def render_week_html(all_data, day_meta, species, points, start):
    return WEEK_TEMPLATE\
        .replace("__DATA__", json.dumps(all_data, ensure_ascii=False))\
        .replace("__META__", json.dumps(day_meta, ensure_ascii=False))\
        .replace("__COORDS__", json.dumps(COORDS, ensure_ascii=False))\
        .replace("__SPECIES__", json.dumps(species, ensure_ascii=False))\
        .replace("__SLOTS__", json.dumps(SLOTS, ensure_ascii=False))\
        .replace("__DATES__", json.dumps(sorted(all_data.keys()), ensure_ascii=False))\
        .replace("__START__", start.isoformat())

WEEK_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>釣果予報(週間) __START__</title>
<style>
 body{font-family:sans-serif;margin:16px;background:#f7f7f7;color:#222}
 h1{font-size:17px;margin:0 0 4px}
 .hdr{font-size:12px;color:#555;margin-bottom:10px;min-height:32px}
 .row{margin:6px 0}
 .row .lab{display:inline-block;width:3.5em;font-size:12px;color:#666;vertical-align:middle}
 .dateribbon{white-space:nowrap;overflow-x:auto;padding-bottom:4px}
 button{margin:2px;padding:5px 9px;border:1px solid #bbb;border-radius:5px;background:#fff;cursor:pointer;font-size:13px}
 button.active{background:#2b6cb0;color:#fff;border-color:#2b6cb0}
 button.datebtn{min-width:52px;text-align:center;line-height:1.25}
 button.datebtn small{display:block;font-size:10px;color:#888}
 button.datebtn.active small{color:#dbeafe}
 button.datebtn.clim{border-style:dashed}
 .legend{display:inline-block;margin-left:10px;font-size:12px;vertical-align:middle}
 .legend i{display:inline-block;width:14px;height:14px;vertical-align:middle;margin:0 2px}
 #map{background:#fff;border:1px solid #ddd;border-radius:8px;max-width:100%}
</style></head><body>
<h1>釣果予報(週間) __START__ 起点</h1>
<div class="hdr" id="dayhdr"></div>
<div class="row"><span class="lab">日付</span><span class="dateribbon" id="dates"></span></div>
<div class="row"><span class="lab">魚種</span><span id="species"></span></div>
<div class="row"><span class="lab">時間帯</span><span id="slots"></span></div>
<div class="row">
 <span class="legend">確率:
  <i style="background:#3b82f6"></i>低
  <i style="background:#22c55e"></i>
  <i style="background:#eab308"></i>
  <i style="background:#f97316"></i>
  <i style="background:#ef4444"></i>高
 </span>
 <span class="legend">（枠が点線の日付＝気候値による目安）</span>
</div>
<svg id="map" width="640" height="560" viewBox="0 0 640 560"></svg>
<div class="hdr">※確率は過去の同条件日での出現割合に基づく傾向値です。実際の釣果を保証するものではありません。</div>
<script>
const DATA=__DATA__, META=__META__, COORDS=__COORDS__, SPECIES=__SPECIES__, SLOTS=__SLOTS__, DATES=__DATES__;
const NS="http://www.w3.org/2000/svg";
const SCALE=80, OFFX=40, OFFY=40, CELL=64;
let curDate=DATES[0], curSp=SPECIES[0], curSlot=SLOTS[0];
const WD=['日','月','火','水','木','金','土'];
function color(p){
  if(p==null) return '#dddddd';
  const stops=[[0,'#3b82f6'],[25,'#22c55e'],[50,'#eab308'],[75,'#f97316'],[100,'#ef4444']];
  for(let i=0;i<stops.length-1;i++){
    const a=stops[i][0], ca=stops[i][1], b=stops[i+1][0], cb=stops[i+1][1];
    if(p>=a&&p<=b) return mix(ca,cb,(p-a)/(b-a));
  }
  return '#ef4444';
}
function mix(c1,c2,t){
  const h=x=>[parseInt(x.slice(1,3),16),parseInt(x.slice(3,5),16),parseInt(x.slice(5,7),16)];
  const a=h(c1), b=h(c2);
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*t)+','+Math.round(a[1]+(b[1]-a[1])*t)+','+Math.round(a[2]+(b[2]-a[2])*t)+')';
}
function textColor(p){ return (p!=null && p>=50) ? '#fff' : '#111'; }
function fmtDate(iso){ const d=new Date(iso+'T00:00:00'); return (d.getMonth()+1)+'/'+d.getDate(); }
function weekday(iso){ return WD[new Date(iso+'T00:00:00').getDay()]; }
function updateHeader(){
  const m=META[curDate]||{};
  const src=(m.wsrc==='気候値')?' ［気候値・目安］':'';
  document.getElementById('dayhdr').innerHTML=
    curDate+'（'+weekday(curDate)+'）'+src+' ／ 推定水温'+m.water+'℃　気温'+(m.air||'-')+
    '　風'+(m.wind_dir||'-')+(m.wind_speed||'')+'m　'+(m.tide||'')+'　'+(m.weather||'')+
    '　月齢'+m.moon_age+'　日の出'+m.sunrise+' 日の入'+m.sunset;
}
function draw(){
  updateHeader();
  const svg=document.getElementById('map');
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  const probs=((DATA[curDate]||{})[curSp]||{})[curSlot]||{};
  for(const p in COORDS){
    const cx=OFFX+COORDS[p][0]*SCALE, cy=OFFY+COORDS[p][1]*SCALE;
    const v=(p in probs)?probs[p]:null;
    const rect=document.createElementNS(NS,'rect');
    rect.setAttribute('x',cx-CELL/2); rect.setAttribute('y',cy-CELL/2);
    rect.setAttribute('width',CELL); rect.setAttribute('height',CELL);
    rect.setAttribute('rx',8); rect.setAttribute('fill',color(v));
    rect.setAttribute('stroke','#333'); rect.setAttribute('stroke-width',1);
    svg.appendChild(rect);
    const t1=document.createElementNS(NS,'text');
    t1.setAttribute('x',cx); t1.setAttribute('y',cy-6); t1.setAttribute('text-anchor','middle');
    t1.setAttribute('font-size','15'); t1.setAttribute('font-weight','bold'); t1.setAttribute('fill',textColor(v));
    t1.textContent=p; svg.appendChild(t1);
    const t2=document.createElementNS(NS,'text');
    t2.setAttribute('x',cx); t2.setAttribute('y',cy+12); t2.setAttribute('text-anchor','middle');
    t2.setAttribute('font-size','12'); t2.setAttribute('fill',textColor(v));
    t2.textContent=(v==null?'-':v+'%'); svg.appendChild(t2);
  }
}
function mkDateRibbon(){
  const box=document.getElementById('dates');
  DATES.forEach(function(iso){
    const b=document.createElement('button');
    b.className='datebtn'+((META[iso]&&META[iso].wsrc==='気候値')?' clim':'')+(iso===curDate?' active':'');
    b.innerHTML=fmtDate(iso)+'<small>'+weekday(iso)+'</small>';
    b.onclick=function(){ curDate=iso; var ch=box.children; for(var i=0;i<ch.length;i++) ch[i].classList.remove('active'); b.classList.add('active'); draw(); };
    box.appendChild(b);
  });
}
function mkButtons(id,items,getCur,setCur){
  const box=document.getElementById(id);
  items.forEach(function(it){
    const b=document.createElement('button');
    b.textContent=it; if(it===getCur()) b.className='active';
    b.onclick=function(){ setCur(it); var ch=box.children; for(var i=0;i<ch.length;i++) ch[i].className=''; b.className='active'; draw(); };
    box.appendChild(b);
  });
}
mkDateRibbon();
mkButtons('species',SPECIES,function(){return curSp;},function(v){curSp=v;});
mkButtons('slots',SLOTS,function(){return curSlot;},function(v){curSlot=v;});
draw();
</script>
</body></html>"""

if __name__ == "__main__":
    main()
