# -*- coding: utf-8 -*-
"""
色分け地図予報: 複数魚種×全時間帯の「いる確率」を計算し、
ポイント配置図の上に色分け表示するHTMLを生成する。

魚種と時間帯をボタンで切り替えて閲覧できる(魚種11×時間帯5=55通り)。

使い方:
    python3 predict_map.py umizuri_clean.csv tomorrow.txt

前提: train_all.py で各魚種のモデル(model_*.txt)を学習済みであること。
出力: forecast_map_<date>.html (ダブルクリックでブラウザ表示)
"""
import sys, math, datetime, json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from astral import LocationInfo
from astral.sun import sun
import ephem

TARGETS = ["クロ","アジ","キス","ヒラメ","セイゴ","カサゴ（アラカブ）",
           "キジハタ","メバル","バリ","マルアジ","アオリイカ"]

LOC=LocationInfo('Fukuoka','Japan','Asia/Tokyo',33.56,130.23)
OFFSHORE_BEARING=45.0
WIND_DIRS={'北':0,'北北東':22.5,'北東':45,'東北東':67.5,'東':90,'東南東':112.5,
    '南東':135,'南南東':157.5,'南':180,'南南西':202.5,'南西':225,'西南西':247.5,
    '西':270,'西北西':292.5,'北西':315,'北北西':337.5}
POINT_ATTR={'A':('tip',0.9,1),'B':('tip',0.9,0),'C':('tip',0.9,0),'D':('tip',1.0,0),
    'E':('tip',0.9,0),'F':('tip',0.9,1),'G':('tip',0.8,1),'H':('tip',0.8,0),
    'I':('tip',0.8,0),'J':('tip',0.85,0),'K':('tip',0.8,0),'L':('tip',0.8,1),
    'M':('corridor',0.6,0),'Q':('corridor',0.6,0),'N':('corridor',0.5,0),'R':('corridor',0.5,0),
    'O':('inner',0.3,1),'P':('inner',0.2,1),'S':('inner',0.35,0),'T':('inner',0.35,0),
    'U':('inner',0.3,1),'V':('inner',0.3,0),'W':('inner',0.25,1)}
COORDS={'A':(0,0),'B':(1,0),'C':(2,0),'D':(4,0),'E':(5,0),'F':(6,0),'G':(0,1),'H':(1,1),'I':(2,1),'J':(4,1),'K':(5,1),'L':(6,1),'M':(2.55,2),'Q':(3.45,2),'N':(2.55,3),'R':(3.45,3),'O':(2.55,4),'S':(3.45,4),'T':(4.35,4),'U':(5.25,4),'P':(2.55,5),'V':(5.25,5),'W':(5.25,6)}
SLOT_CENTER={'6-9':7.5,'9-12':10.5,'12-15':13.5,'15-18':16.5,'18-20':19.0}
SLOTS=list(SLOT_CENTER.keys())
FEATURES_NUM=["water_temp","air_temp","wind_speed","center_h","month","doy_sin","doy_cos",
    "to_high_h","to_low_h","moon_age","near_sunrise","near_sunset","onshore","onshore_strength","exposure","corner"]
FEATURES_CAT=["tide_type","time_slot","point","zone","wind_dir"]

def model_filename(target):
    # 日本語ファイル名はWindowsのLightGBMが扱えないため、TARGETS内の
    # 並び順(インデックス)で英数字ファイル名にする(train_all.pyと同一規則)。
    return f"model_{TARGETS.index(target):02d}.txt"

def read_conditions(path):
    cond={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            if '=' in line:
                k,v=line.split('=',1); cond[k.strip()]=v.split('#')[0].strip()
    return cond

def estimate_water_temp(recent,weather):
    r=[t for t in recent if t is not None]
    if not r: return None
    base=r[-1]; diffs=[r[i+1]-r[i] for i in range(max(0,len(r)-3),len(r)-1)]
    daily=float(np.mean(diffs)) if diffs else 0.0
    adj={'晴れ':0.4,'快晴':0.5,'曇り':0.0,'雨':-0.4,'大雨':-0.7}
    return round(base+daily+adj.get(weather,0.0),1)

def to_hours(hhmm):
    try: h,m=hhmm.split(':'); return int(h)+int(m)/60
    except: return None
def nearest_tide(center,s):
    if not s: return None
    ts=[to_hours(t) for t in s.split('|') if to_hours(t) is not None]
    return min([center-t for t in ts],key=abs) if ts else None

def build_features(cond, water, points):
    d=datetime.date.fromisoformat(cond['date'])
    pn=ephem.previous_new_moon(d); m_age=float(ephem.Date(d)-pn)
    s=sun(LOC.observer,date=d,tzinfo='Asia/Tokyo')
    sr=s['sunrise'].hour+s['sunrise'].minute/60; ss=s['sunset'].hour+s['sunset'].minute/60
    doy=d.timetuple().tm_yday
    wdir=cond.get('wind_dir'); wspd=float(cond.get('wind_speed',0) or 0)
    wdeg=WIND_DIRS.get(wdir); onshore=math.cos(math.radians(wdeg-OFFSHORE_BEARING)) if wdeg is not None else 0.0
    rows=[]
    for slot,center in SLOT_CENTER.items():
        for p in points:
            zone,exp,cor=POINT_ATTR[p]
            rows.append(dict(water_temp=water,air_temp=float(cond.get('air_temp') or water),
                wind_speed=wspd,center_h=center,month=d.month,
                doy_sin=math.sin(2*math.pi*doy/366),doy_cos=math.cos(2*math.pi*doy/366),
                to_high_h=nearest_tide(center,cond.get('high_tides','')),
                to_low_h=nearest_tide(center,cond.get('low_tides','')),moon_age=m_age,
                near_sunrise=max(0,1-abs(center-sr)/3),near_sunset=max(0,1-abs(center-ss)/3),
                onshore=onshore,onshore_strength=onshore*wspd,exposure=exp,corner=cor,
                tide_type=cond.get('tide_type'),time_slot=slot,point=p,zone=zone,wind_dir=wdir))
    return pd.DataFrame(rows), dict(moon_age=round(m_age,1),sunrise=round(sr,1),sunset=round(ss,1))

def main(clean_csv, cond_path):
    cond=read_conditions(cond_path)
    recent=[float(x) for x in cond.get('recent_water','').split(',') if x.strip()]
    water=estimate_water_temp(recent,cond.get('weather','曇り'))
    if water is None:
        print("recent_water が必要です",file=sys.stderr); return
    df=pd.read_csv(clean_csv)
    points=sorted([p for p in df['point'].dropna().unique() if p in POINT_ATTR])
    X,astro=build_features(cond,water,points)
    for c in FEATURES_CAT: X[c]=X[c].astype('category')

    # 各魚種のモデルで予測。probs[species][slot][point] = 確率
    probs={}
    trained=[]
    for t in TARGETS:
        mf=model_filename(t)
        if not os.path.exists(mf):
            print(f"  モデル未学習でスキップ: {t} ({mf})",file=sys.stderr); continue
        model=lgb.Booster(model_file=mf)
        X['_p']=model.predict(X[FEATURES_NUM+FEATURES_CAT])
        d={}
        for slot in SLOTS:
            sub=X[X['time_slot']==slot]
            d[slot]={r['point']:round(float(r['_p'])*100,1) for _,r in sub.iterrows()}
        probs[t]=d; trained.append(t)
    X.drop(columns=['_p'],inplace=True,errors='ignore')

    html=render_html(cond,water,astro,probs,trained,points)
    out=f"forecast_map_{cond['date']}.html"
    with open(out,'w',encoding='utf-8') as f: f.write(html)
    print(f"生成: {out}  (魚種{len(trained)} × 時間帯{len(SLOTS)})")
    print(f"ダブルクリックでブラウザ表示。魚種と時間帯のボタンで切り替え。")

def render_html(cond,water,astro,probs,species,points):
    data_json=json.dumps(probs,ensure_ascii=False)
    coords_json=json.dumps(COORDS,ensure_ascii=False)
    species_json=json.dumps(species,ensure_ascii=False)
    slots_json=json.dumps(SLOTS,ensure_ascii=False)
    header=(f"{cond['date']} 釣果予報 / 推定水温{water}℃ 気温{cond.get('air_temp')} "
            f"風{cond.get('wind_dir')}{cond.get('wind_speed')}m {cond.get('tide_type')} "
            f"月齢{astro['moon_age']} 日の出{astro['sunrise']} 日の入{astro['sunset']}")
    return TEMPLATE.replace('__DATA__',data_json).replace('__COORDS__',coords_json)\
        .replace('__SPECIES__',species_json).replace('__SLOTS__',slots_json)\
        .replace('__HEADER__',header).replace('__DATE__',cond['date'])

TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>釣果予報 __DATE__</title>
<style>
 body{font-family:sans-serif;margin:16px;background:#f7f7f7;color:#222}
 h1{font-size:16px}
 .hdr{font-size:12px;color:#555;margin-bottom:10px}
 .controls{margin:8px 0}
 button{margin:2px;padding:5px 9px;border:1px solid #bbb;border-radius:5px;background:#fff;cursor:pointer;font-size:13px}
 button.active{background:#2b6cb0;color:#fff;border-color:#2b6cb0}
 .legend{display:inline-block;margin-left:10px;font-size:12px;vertical-align:middle}
 .legend i{display:inline-block;width:14px;height:14px;vertical-align:middle;margin:0 2px}
 #map{background:#fff;border:1px solid #ddd;border-radius:8px}
</style></head><body>
<h1>釣果予報 __DATE__</h1>
<div class="hdr">__HEADER__</div>
<div class="controls" id="species"></div>
<div class="controls" id="slots"></div>
<div class="controls">
 <span class="legend">確率:
  <i style="background:#3b82f6"></i>低
  <i style="background:#22c55e"></i>
  <i style="background:#eab308"></i>
  <i style="background:#f97316"></i>
  <i style="background:#ef4444"></i>高
 </span>
</div>
<svg id="map" width="640" height="560"></svg>
<div class="hdr">※確率は過去の同条件日での出現割合に基づく傾向値です。実際の釣果を保証するものではありません。</div>
<script>
const DATA=__DATA__, COORDS=__COORDS__, SPECIES=__SPECIES__, SLOTS=__SLOTS__;
const NS="http://www.w3.org/2000/svg";
const SCALE=80, OFFX=40, OFFY=40, CELL=64; // ピクセル配置
let curSp=SPECIES[0], curSlot=SLOTS[0];
function color(p){
  if(p==null) return '#dddddd';
  const stops=[[0,'#3b82f6'],[25,'#22c55e'],[50,'#eab308'],[75,'#f97316'],[100,'#ef4444']];
  for(let i=0;i<stops.length-1;i++){
    const a=stops[i][0], ca=stops[i][1], b=stops[i+1][0], cb=stops[i+1][1];
    if(p>=a&&p<=b){ return mix(ca,cb,(p-a)/(b-a)); }
  }
  return '#ef4444';
}
function mix(c1,c2,t){
  const h=x=>[parseInt(x.slice(1,3),16),parseInt(x.slice(3,5),16),parseInt(x.slice(5,7),16)];
  const a=h(c1), b=h(c2);
  const r=Math.round(a[0]+(b[0]-a[0])*t), g=Math.round(a[1]+(b[1]-a[1])*t), bl=Math.round(a[2]+(b[2]-a[2])*t);
  return 'rgb('+r+','+g+','+bl+')';
}
function textColor(p){ return (p!=null && p>=50) ? '#fff' : '#111'; }
function draw(){
  const svg=document.getElementById('map');
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  const probs=(DATA[curSp]||{})[curSlot]||{};
  for(const p in COORDS){
    const cx=OFFX + COORDS[p][0]*SCALE;
    const cy=OFFY + COORDS[p][1]*SCALE;
    const v=(p in probs)? probs[p] : null;
    const rect=document.createElementNS(NS,'rect');
    rect.setAttribute('x', cx - CELL/2);
    rect.setAttribute('y', cy - CELL/2);
    rect.setAttribute('width', CELL);
    rect.setAttribute('height', CELL);
    rect.setAttribute('rx', 8);
    rect.setAttribute('fill', color(v));
    rect.setAttribute('stroke', '#333');
    rect.setAttribute('stroke-width', 1);
    svg.appendChild(rect);
    const t1=document.createElementNS(NS,'text');
    t1.setAttribute('x', cx); t1.setAttribute('y', cy-6);
    t1.setAttribute('text-anchor','middle');
    t1.setAttribute('font-size','15'); t1.setAttribute('font-weight','bold');
    t1.setAttribute('fill', textColor(v));
    t1.textContent=p;
    svg.appendChild(t1);
    const t2=document.createElementNS(NS,'text');
    t2.setAttribute('x', cx); t2.setAttribute('y', cy+12);
    t2.setAttribute('text-anchor','middle');
    t2.setAttribute('font-size','12');
    t2.setAttribute('fill', textColor(v));
    t2.textContent=(v==null?'-':v+'%');
    svg.appendChild(t2);
  }
}
function mkButtons(id,items,cur,cb){
  const box=document.getElementById(id);
  items.forEach(function(it){
    const b=document.createElement('button'); b.textContent=it;
    if(it===cur) b.className='active';
    b.onclick=function(){ cb(it); var ch=box.children; for(var i=0;i<ch.length;i++) ch[i].className=''; b.className='active'; };
    box.appendChild(b);
  });
}
mkButtons('species',SPECIES,curSp,function(v){curSp=v;draw();});
mkButtons('slots',SLOTS,curSlot,function(v){curSlot=v;draw();});
draw();
</script>
</body></html>"""

if __name__ == "__main__":
    clean=sys.argv[1] if len(sys.argv)>1 else "umizuri_clean.csv"
    cond=sys.argv[2] if len(sys.argv)>2 else "tomorrow.txt"
    main(clean, cond)
