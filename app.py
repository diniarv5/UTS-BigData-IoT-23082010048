import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
st.set_page_config(page_title='DANA Dashboard', layout='wide')

# ─────────────────────────────
# LOAD DATA (FIX FINAL PATH)
# ─────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('/content/df_combined_final.csv')
        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        df['sentimen'] = df['sentimen'].str.upper().str.strip()
        df = df.dropna(subset=['at'])
        return df
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Dataset tidak ditemukan / gagal dibaca.")
    st.stop()

# ─────────────────────────────
# SIDEBAR FILTER
# ─────────────────────────────
st.sidebar.title("📊 Filter")

tahun = st.sidebar.multiselect(
    "Pilih Tahun",
    sorted(df['at'].dt.year.unique()),
    default=sorted(df['at'].dt.year.unique())
)

granularity = st.sidebar.selectbox(
    "Time Analysis",
    ["Weekly", "Monthly"]
)

df = df[df['at'].dt.year.isin(tahun)]

# ─────────────────────────────
# HEADER
# ─────────────────────────────
st.title("📊 DANA Sentiment Monitoring Dashboard")

# ─────────────────────────────
# KPI (REAL DATA)
# ─────────────────────────────
total = len(df)
avg_rating = df['score'].mean()
pos_pct = (df['sentimen']=='POSITIVE').mean()*100
neg_pct = (df['sentimen']=='NEGATIVE').mean()*100
crit_pct = (df['score']<=2).mean()*100

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Reviews", f"{total:,}")
c2.metric("Avg Rating", f"{avg_rating:.2f}")
c3.metric("Positive %", f"{pos_pct:.1f}%")
c4.metric("Critical %", f"{crit_pct:.1f}%")

# ─────────────────────────────
# SENTIMENT DISTRIBUTION
# ─────────────────────────────
st.subheader("📊 Sentiment Distribution")

sent_counts = df['sentimen'].value_counts().reset_index()
sent_counts.columns = ['sentimen','count']

fig = px.pie(
    sent_counts,
    values='count',
    names='sentimen',
    hole=0.5
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
# TREND ANALYSIS
# ─────────────────────────────
st.subheader("📈 Trend Sentiment")

temp = df.copy()

if granularity == "Weekly":
    temp['time'] = temp['at'].dt.to_period('W').apply(lambda x: x.start_time)
else:
    temp['time'] = temp['at'].dt.to_period('M').apply(lambda x: x.start_time)

trend = temp.groupby(['time','sentimen']).size().reset_index(name='count')

fig2 = px.line(
    trend,
    x='time',
    y='count',
    color='sentimen',
    markers=True
)
st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────
# KEYWORD ANALYSIS (NEGATIVE)
# ─────────────────────────────
st.subheader("🔥 Top Negative Keywords")

neg_text = ' '.join(df[df['sentimen']=='NEGATIVE']['content'].astype(str)).lower()

words = re.findall(r'[a-z]{3,}', neg_text)

# stopwords sederhana
stopwords = {'dan','yang','di','ke','dari','ini','itu','saya','tidak','ada','untuk','dana'}
words = [w for w in words if w not in stopwords]

freq = Counter(words)
top10 = dict(freq.most_common(10))

fig3 = px.bar(
    x=list(top10.values()),
    y=list(top10.keys()),
    orientation='h'
)
st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────
# WORDCLOUD
# ─────────────────────────────
st.subheader("☁️ WordCloud")

if neg_text.strip():
    wc = WordCloud(width=800, height=400).generate(neg_text)
    fig_wc, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')
    st.pyplot(fig_wc)

# ─────────────────────────────
# CRITICAL KEYWORD TRACKING
# ─────────────────────────────
st.subheader("🚨 Critical Keyword Tracking")

keywords = ['saldo','hilang','premium','login','otp','gagal']

data_kw = []

for kw in keywords:
    count = df['content'].str.contains(kw, case=False, na=False).sum()
    data_kw.append((kw, count))

kw_df = pd.DataFrame(data_kw, columns=['Keyword','Count'])

fig4 = px.bar(kw_df, x='Count', y='Keyword', orientation='h')
st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────
# KEYWORD TREND (PREDICTIVE)
# ─────────────────────────────
st.subheader("📈 Keyword Trend (Predictive Insight)")

temp_kw = df.copy()
temp_kw['time'] = temp_kw['at'].dt.to_period('W').apply(lambda x: x.start_time)

rows = []

for kw in ['saldo','hilang','premium']:
    mask = temp_kw['content'].str.contains(kw, case=False, na=False)
    d = temp_kw[mask].groupby('time').size().reset_index(name='count')
    d['keyword'] = kw
    rows.append(d)

kw_trend = pd.concat(rows)

fig5 = px.line(
    kw_trend,
    x='time',
    y='count',
    color='keyword',
    markers=True
)
st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────
# CHURN ANALYSIS
# ─────────────────────────────
st.subheader("🚨 Churn Risk Users")

if 'userName' in df.columns:
    churn = df.groupby('userName').agg(
        neg=('sentimen', lambda x: (x=='NEGATIVE').sum()),
        avg=('score','mean')
    )

    churn = churn[(churn['neg']>=2) & (churn['avg']<=2.5)]

    st.metric("High Risk Users", len(churn))
    st.dataframe(churn.sort_values('neg', ascending=False).head(10))

# ─────────────────────────────
# RAW DATA
# ─────────────────────────────
with st.expander("📂 Raw Data"):
    st.dataframe(df.head(200))
