"""
DANA App Sentiment & Health Monitoring Dashboard
================================================
Kompatibel : Streamlit >= 1.20, Google Colab, lokal, Streamlit Cloud
Data       : df_combined_final.csv  (53.000 ulasan · 13 kolom)
Run        : streamlit run app.py
"""
import os, re, io
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ══════════════════════════════════════════════
#  0. PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="DANA Sentiment Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  1. CSS — dark / light + responsive
# ══════════════════════════════════════════════
st.markdown("""
<style>
:root{--blue:#1C85C7;--blue-d:#0D5A96;--green:#22C55E;--red:#EF4444;
  --amber:#F59E0B;--purple:#8B5CF6;--r:14px;--gap:1rem}
#MainMenu,footer,header{visibility:hidden}
.stApp{--bg:#F0F4F9;--card:#FFFFFF;--txt:#0D1B2A;--mute:#5A6B7E;
  --bdr:#DDE3EC;--side:#0D5A96}
@media(prefers-color-scheme:dark){
  .stApp{--bg:#0D1117;--card:#161B22;--txt:#E6EDF3;--mute:#8B949E;
    --bdr:#30363D;--side:#0A1628}}
.stApp{background:var(--bg)!important}
[data-testid="stSidebar"]{background:var(--side)!important}
[data-testid="stSidebar"] *{color:#fff!important}
[data-testid="stSidebar"] label{font-size:.8rem!important;font-weight:600}
.dana-banner{background:linear-gradient(135deg,var(--blue) 0%,var(--blue-d) 100%);
  border-radius:var(--r);padding:1.4rem 2rem;display:flex;align-items:center;
  justify-content:space-between;flex-wrap:wrap;gap:.6rem;margin-bottom:1.2rem}
.dana-banner h1{color:#fff;font-size:clamp(1.1rem,2.6vw,1.8rem);
  font-weight:800;margin:0;line-height:1.25}
.dana-banner .sub{color:rgba(255,255,255,.75);
  font-size:clamp(.68rem,1.3vw,.82rem);margin-top:.3rem}
.badge-live{background:#22C55E;color:#fff;font-size:.7rem;font-weight:700;
  padding:.3rem .85rem;border-radius:20px;white-space:nowrap;letter-spacing:.05em}
.sec{font-size:clamp(.88rem,1.7vw,1.05rem);font-weight:700;color:var(--txt);
  padding-bottom:.35rem;border-bottom:2.5px solid var(--blue);
  margin:1.4rem 0 .75rem;display:flex;align-items:center;gap:.45rem}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
  gap:var(--gap);margin-bottom:.5rem}
.kcard{background:var(--card);border:1px solid var(--bdr);
  border-radius:var(--r);padding:1rem 1.1rem;position:relative;overflow:hidden}
.kcard::after{content:'';position:absolute;top:0;left:0;right:0;height:4px;
  border-radius:var(--r) var(--r) 0 0}
.kcard.tot::after{background:var(--purple)}.kcard.pos::after{background:var(--green)}
.kcard.neg::after{background:var(--red)}.kcard.neu::after{background:var(--amber)}
.kcard.rat::after{background:var(--blue)}.kcard.crt::after{background:#F97316}
.klabel{font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.07em;color:var(--mute);margin-bottom:.35rem}
.kvalue{font-size:clamp(1.25rem,2.3vw,1.8rem);font-weight:800;line-height:1;margin-bottom:.2rem}
.kcard.tot .kvalue{color:var(--purple)}.kcard.pos .kvalue{color:var(--green)}
.kcard.neg .kvalue{color:var(--red)}.kcard.neu .kvalue{color:var(--amber)}
.kcard.rat .kvalue{color:var(--blue)}.kcard.crt .kvalue{color:#F97316}
.ktarget{font-size:.67rem;color:var(--mute)}
.kwrow{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:.8rem}
.kwpill{background:var(--card);border:1px solid var(--bdr);border-radius:10px;
  padding:.5rem .8rem;display:flex;flex-direction:column;align-items:center;
  gap:.05rem;min-width:88px;flex:1}
.kwword{font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.05em;color:var(--mute)}
.kwval{font-size:1.25rem;font-weight:800}
.churn-box{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);
  border-radius:10px;padding:.75rem 1rem;margin-bottom:.65rem;
  font-size:.85rem;color:var(--txt)}
@media(max-width:768px){.kpi-grid{grid-template-columns:repeat(2,1fr)}
  .dana-banner{padding:1rem 1.15rem}}
@media(max-width:420px){.kpi-grid{grid-template-columns:1fr 1fr}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  2. DATA LOADING — multi-path fallback
# ══════════════════════════════════════════════
FILENAME = "df_combined_final.csv"
SEARCH_PATHS = [
    FILENAME,
    Path(__file__).parent / FILENAME,
    f"/content/{FILENAME}",
    f"/mnt/user-data/uploads/{FILENAME}",
    Path.home() / FILENAME,
]

@st.cache_data(show_spinner="Memuat dataset…")
def load_data():
    for p in SEARCH_PATHS:
        if Path(p).exists():
            df = pd.read_csv(str(p))
            df["at"] = pd.to_datetime(df["at"], errors="coerce")
            df["sentimen"] = df["sentimen"].str.strip().str.lower()
            df["content"]  = df["content"].fillna("")
            return df, str(p)
    return pd.DataFrame(), None

df, found_path = load_data()

# ══════════════════════════════════════════════
#  3. SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='background:rgba(255,255,255,.12);border-radius:10px;
    padding:.85rem;margin-bottom:1rem;text-align:center;'>
      <div style='font-size:1.6rem;font-weight:900;letter-spacing:.04em'>📊 DANA</div>
      <div style='font-size:.7rem;opacity:.75;margin-top:.2rem'>
        Sentiment &amp; Health Monitoring
      </div>
    </div>""", unsafe_allow_html=True)

    if df.empty:
        st.error(
            "❌ File tidak ditemukan.\n\n"
            "Letakkan `df_combined_final.csv` di folder yang **sama** dengan `app.py`, "
            "lalu jalankan ulang:\n```\nstreamlit run app.py\n```"
        )
        st.stop()

    st.markdown("**🔍 Filter Dashboard**")

    min_d = df["at"].min().date()
    max_d = df["at"].max().date()
    date_range = st.date_input("Periode", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)

    sent_opts = sorted(df["sentimen"].unique())
    sel_sent  = st.multiselect("Sentimen", sent_opts, default=sent_opts)

    rat_opts = sorted(df["score"].unique())
    sel_rat  = st.multiselect("Rating ★", rat_opts, default=rat_opts)

    d0 = date_range[0] if isinstance(date_range,(list,tuple)) and len(date_range)>0 else min_d
    d1 = date_range[1] if isinstance(date_range,(list,tuple)) and len(date_range)>1 else max_d

    mask = (
        (df["at"].dt.date >= d0) & (df["at"].dt.date <= d1) &
        (df["sentimen"].isin(sel_sent or sent_opts)) &
        (df["score"].isin(sel_rat   or rat_opts))
    )
    fdf = df.loc[mask].copy()

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:.72rem;opacity:.65;text-align:center;'>"
        f"✅ {len(fdf):,} dari {len(df):,} ulasan<br>"
        f"📂 {Path(found_path).name}</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════
#  4. HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class="dana-banner">
  <div>
    <h1>📊 DANA App Sentiment &amp; Health Monitoring</h1>
    <div class="sub">Analisis Ulasan Pengguna · Google Play Store · 53.000 Ulasan · 2024</div>
  </div>
  <div class="badge-live">● LIVE</div>
</div>
""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("⚠️ Tidak ada data sesuai filter. Sesuaikan pilihan di sidebar.")
    st.stop()

# ══════════════════════════════════════════════
#  5. STATS
# ══════════════════════════════════════════════
total    = len(fdf)
pos_n    = (fdf["sentimen"]=="positif").sum()
neg_n    = (fdf["sentimen"]=="negatif").sum()
neu_n    = (fdf["sentimen"]=="netral").sum()
pos_pct  = pos_n/total*100
neg_pct  = neg_n/total*100
neu_pct  = neu_n/total*100
avg_rat  = fdf["score"].mean()
crit_pct = (fdf["score"]<=2).mean()*100
saldo_n  = int(fdf["is_saldo_hilang"].sum()) if "is_saldo_hilang" in fdf else 0
prem_n   = int(fdf["is_premium"].sum())      if "is_premium"      in fdf else 0

# ══════════════════════════════════════════════
#  6. KPI CARDS
# ══════════════════════════════════════════════
st.markdown('<div class="sec">🎯 Key Performance Indicators</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-grid">
  <div class="kcard tot"><div class="klabel">Total Ulasan</div>
    <div class="kvalue">{total:,}</div><div class="ktarget">Dataset aktif</div></div>
  <div class="kcard pos"><div class="klabel">Sentimen Positif</div>
    <div class="kvalue">{pos_pct:.1f}%</div>
    <div class="ktarget">Target &gt;80% · {pos_n:,} ulasan</div></div>
  <div class="kcard neg"><div class="klabel">Sentimen Negatif</div>
    <div class="kvalue">{neg_pct:.1f}%</div>
    <div class="ktarget">Target &lt;15% · {neg_n:,} ulasan</div></div>
  <div class="kcard neu"><div class="klabel">Sentimen Netral</div>
    <div class="kvalue">{neu_pct:.1f}%</div>
    <div class="ktarget">{neu_n:,} ulasan</div></div>
  <div class="kcard rat"><div class="klabel">Avg Rating</div>
    <div class="kvalue">{avg_rat:.2f} ★</div>
    <div class="ktarget">Target &gt;4.0 / 5.0</div></div>
  <div class="kcard crt"><div class="klabel">Critical Rate</div>
    <div class="kvalue">{crit_pct:.1f}%</div>
    <div class="ktarget">Target &lt;10% · rating 1–2★</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  7. DONUT + WEEKLY TREND
# ══════════════════════════════════════════════
st.markdown('<div class="sec">📈 Distribusi &amp; Tren Sentimen</div>', unsafe_allow_html=True)
cA1, cA2 = st.columns([1,1.7], gap="medium")

_cm = {"Positif":"#22C55E","Negatif":"#EF4444","Netral":"#F59E0B"}
_lm = {"positif":"Positif","negatif":"Negatif","netral":"Netral"}

_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               font=dict(family="Inter,Arial,sans-serif"),
               margin=dict(t=44,b=20,l=10,r=10))

with cA1:
    pie_df = pd.DataFrame({"S":["Positif","Negatif","Netral"],"N":[pos_n,neg_n,neu_n]})
    fig = px.pie(pie_df,names="S",values="N",hole=0.52,color="S",
                 color_discrete_map=_cm,title="Distribusi Sentimen")
    fig.update_traces(textposition="outside",textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:,} ulasan · %{percent}<extra></extra>")
    fig.update_layout(**_LAYOUT,showlegend=False,title=dict(x=0.5,font=dict(size=14,weight=700)))
    st.plotly_chart(fig, use_container_width=True)

with cA2:
    fdf["_wk"] = pd.to_datetime(fdf["week"], errors="coerce")
    tr = fdf.groupby(["_wk","sentimen"]).size().reset_index(name="n")
    tr["label"] = tr["sentimen"].map(_lm)
    fig = px.line(tr,x="_wk",y="n",color="label",color_discrete_map=_cm,
                  markers=True,title="Tren Ulasan per Minggu",
                  labels={"_wk":"Minggu","n":"Jumlah","label":"Sentimen"})
    fig.update_traces(line_width=2.5,marker_size=6)
    fig.update_layout(**_LAYOUT,
        xaxis=dict(showgrid=False,tickformat="%d %b"),
        yaxis=dict(showgrid=True,gridcolor="rgba(150,150,150,.15)"),
        legend=dict(orientation="h",y=-0.2,title=""),hovermode="x unified",
        title=dict(x=0.5,font=dict(size=14,weight=700)))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
#  8. STACKED BAR + DAY-OF-WEEK
# ══════════════════════════════════════════════
st.markdown('<div class="sec">📊 Rating &amp; Pola Hari</div>', unsafe_allow_html=True)
cB1, cB2 = st.columns([1.3,1], gap="medium")

with cB1:
    sb = fdf.groupby(["score","sentimen"]).size().reset_index(name="n")
    sb["label"] = sb["sentimen"].map(_lm)
    fig = px.bar(sb,x="score",y="n",color="label",color_discrete_map=_cm,
                 barmode="stack",title="Distribusi Sentimen per Rating",
                 labels={"score":"Rating ★","n":"Jumlah","label":"Sentimen"},
                 category_orders={"score":[1,2,3,4,5]})
    fig.update_layout(**_LAYOUT,
        xaxis=dict(showgrid=False,tickmode="array",tickvals=[1,2,3,4,5],
                   ticktext=["1★","2★","3★","4★","5★"]),
        yaxis=dict(showgrid=True,gridcolor="rgba(150,150,150,.15)"),
        legend=dict(orientation="h",y=-0.2,title=""),bargap=0.22,
        title=dict(x=0.5,font=dict(size=14,weight=700)))
    st.plotly_chart(fig, use_container_width=True)

with cB2:
    if "day_of_week" in fdf.columns:
        dn = {0:"Sen",1:"Sel",2:"Rab",3:"Kam",4:"Jum",5:"Sab",6:"Min"}
        dow = fdf.groupby(["day_of_week","sentimen"]).size().reset_index(name="n")
        dow["day"]   = dow["day_of_week"].map(dn)
        dow["label"] = dow["sentimen"].map(_lm)
        fig = px.bar(dow,x="day",y="n",color="label",color_discrete_map=_cm,
                     barmode="group",title="Ulasan per Hari",
                     labels={"day":"Hari","n":"Jumlah","label":"Sentimen"},
                     category_orders={"day":["Sen","Sel","Rab","Kam","Jum","Sab","Min"]})
        fig.update_layout(**_LAYOUT,
            xaxis_showgrid=False,
            yaxis=dict(showgrid=True,gridcolor="rgba(150,150,150,.15)"),
            legend=dict(orientation="h",y=-0.2,title=""),bargap=0.18,
            title=dict(x=0.5,font=dict(size=14,weight=700)))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
#  9. TOP-10 KEYWORD BARS
# ══════════════════════════════════════════════
st.markdown('<div class="sec">🔤 Top 10 Kata Kunci</div>', unsafe_allow_html=True)

STOPS = {
    "yang","dan","di","ke","dari","dengan","ini","itu","saya","tidak","ada","untuk",
    "sudah","bisa","masih","jadi","tapi","atau","juga","hanya","kalau","aja","ya",
    "kok","yg","sy","nya","lg","tp","gak","ga","gk","ngga","nggak","lagi","mau",
    "klo","nih","sih","deh","lah","karna","karena","via","app","apk","aplikasi",
    "dana","sangat","banget","sekali","memang","emang","aku","kamu","mereka",
    "bukan","jangan","cuma","saja","pun","pula","agar","supaya","walau",
    "meski","karena","hari","bulan","tahun","jam","menit","mu","ku",
}

def top_words(texts, n=10):
    tokens = [w for w in re.findall(r"[a-z]{3,}"," ".join(texts).lower()) if w not in STOPS]
    return pd.DataFrame(Counter(tokens).most_common(n),columns=["Kata","Frekuensi"])

cC1,cC2 = st.columns(2,gap="medium")
pos_texts = fdf[fdf["sentimen"]=="positif"]["content"].tolist()
neg_texts = fdf[fdf["sentimen"]=="negatif"]["content"].tolist()

with cC1:
    pkdf = top_words(pos_texts)
    fig = px.bar(pkdf,x="Frekuensi",y="Kata",orientation="h",
                 color_discrete_sequence=["#22C55E"],title="Top 10 — Positif")
    fig.update_layout(**_LAYOUT,
        yaxis=dict(categoryorder="total ascending"),
        xaxis=dict(showgrid=True,gridcolor="rgba(150,150,150,.15)"),
        title=dict(x=0.5,font=dict(size=14,weight=700)))
    st.plotly_chart(fig,use_container_width=True)

with cC2:
    nkdf = top_words(neg_texts)
    fig = px.bar(nkdf,x="Frekuensi",y="Kata",orientation="h",
                 color_discrete_sequence=["#EF4444"],title="Top 10 — Negatif")
    fig.update_layout(**_LAYOUT,
        yaxis=dict(categoryorder="total ascending"),
        xaxis=dict(showgrid=True,gridcolor="rgba(150,150,150,.15)"),
        title=dict(x=0.5,font=dict(size=14,weight=700)))
    st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════
# 10. WORD CLOUD + CHURN TABLE
# ══════════════════════════════════════════════
st.markdown('<div class="sec">☁️ Word Cloud &amp; 🚨 Churn Risk Users</div>', unsafe_allow_html=True)
cD1,cD2 = st.columns([1.1,1],gap="medium")

with cD1:
    freq = Counter(w for w in re.findall(r"[a-z]{3,}"," ".join(neg_texts).lower())
                   if w not in STOPS)
    wc = WordCloud(width=720,height=380,background_color="white",
                   colormap="RdYlGn_r",max_words=90,prefer_horizontal=0.78,
                   min_font_size=9,max_font_size=95,collocations=False
                   ).generate_from_frequencies(freq)
    fig_wc,ax = plt.subplots(figsize=(7.2,3.8))
    ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
    fig_wc.patch.set_alpha(0); plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig_wc.savefig(buf,format="png",dpi=150,bbox_inches="tight",transparent=True)
    buf.seek(0)
    st.image(buf,caption="Word Cloud — Kata Kunci Ulasan Negatif",use_container_width=True)
    plt.close(fig_wc)

with cD2:
    churn_df = (
        fdf[fdf["sentimen"]=="negatif"]
        .groupby("userName")
        .agg(neg_count=("sentimen","count"),avg_rating=("score","mean"))
        .reset_index()
        .query("neg_count >= 2")
        .sort_values("neg_count",ascending=False)
        .head(10)
    )
    churn_total = int(
        fdf[fdf["sentimen"]=="negatif"]
        .groupby("userName")["sentimen"]
        .count()
        .ge(2).sum()
    )
    churn_df["avg_rating"] = churn_df["avg_rating"].round(1)
    churn_df.columns = ["Pengguna","Review Negatif","Avg Rating"]
    st.markdown(
        f'<div class="churn-box">🚨 <b>{churn_total:,} pengguna</b> berisiko churn tinggi '
        f'(≥ 2 ulasan negatif)</div>',unsafe_allow_html=True)
    st.dataframe(churn_df.reset_index(drop=True),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════
# 11. CRITICAL ISSUE TRACKING + KPI 6 & 7
# ══════════════════════════════════════════════
st.markdown('<div class="sec">⚠️ Critical Issue &amp; Keyword Tracking</div>', unsafe_allow_html=True)

KW_LIST = {
    "saldo":   (r"\bsaldo\b","#EF4444"),
    "hilang":  (r"\bhilang\b","#EF4444"),
    "premium": (r"\bpremium\b","#F97316"),
    "login":   (r"\blogin\b","#F59E0B"),
    "otp":     (r"\botp\b","#F59E0B"),
    "upgrade": (r"\bupgrade\b","#60A5FA"),
    "akun":    (r"\bakun\b","#60A5FA"),
    "gagal":   (r"\bgagal\b","#FB923C"),
    "keamanan":(r"\bkeamanan\b","#A78BFA"),
    "cs":      (r"\bcs\b","#94A3B8"),
}
pills = "".join([
    f'<div class="kwpill"><div class="kwword">{k}</div>'
    f'<div class="kwval" style="color:{c}">'
    f'{fdf["content"].str.contains(pat,case=False,na=False).sum():,}'
    f'</div></div>'
    for k,(pat,c) in KW_LIST.items()
])
st.markdown(f'<div class="kwrow">{pills}</div>',unsafe_allow_html=True)

cE1,cE2 = st.columns(2,gap="medium")
with cE1:
    st.info(f"🚨 **'saldo hilang'** terdeteksi dalam **{saldo_n:,}** ulasan\n\n"
            f"*(kolom `is_saldo_hilang` — pre-computed)*")
with cE2:
    st.warning(f"🛠️ **Keluhan 'premium'** terdeteksi dalam **{prem_n:,}** ulasan\n\n"
               f"*(kolom `is_premium` — pre-computed)*")

# ══════════════════════════════════════════════
# 12. RAW DATA EXPLORER
# ══════════════════════════════════════════════
with st.expander("🗂️ Eksplorasi Data Mentah"):
    disp = fdf[["userName","score","at","sentimen","content"]].copy()
    disp["at"] = disp["at"].dt.strftime("%Y-%m-%d")
    disp.columns = ["Pengguna","Rating","Tanggal","Sentimen","Ulasan"]
    st.dataframe(disp.head(500),use_container_width=True,hide_index=True)
    st.caption(f"500 dari {total:,} baris (setelah filter).")

# ══════════════════════════════════════════════
# 13. FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 .5rem;font-size:.72rem;
color:var(--mute);border-top:1px solid var(--bdr);margin-top:1rem;'>
  DANA Sentiment &amp; Health Monitoring · Streamlit · Plotly · WordCloud · Pandas<br>
  Sumber: <code>df_combined_final.csv</code> · 53.000 ulasan · Google Play Store 2024<br>
  UPN "Veteran" Jawa Timur — Big Data &amp; IoT ETS 2025/2026
</div>
""", unsafe_allow_html=True)
