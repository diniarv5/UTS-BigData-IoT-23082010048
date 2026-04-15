import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image
from collections import Counter

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title='DANA App Monitoring',
    page_icon='📊',
    layout='wide'
)

DANA_BLUE = '#1C85C7'

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'df_combined_final.csv')
LOGO_PATH = os.path.join(BASE_DIR, 'logo dana.png')

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        # Clean kolom
        df['sentimen'] = df['sentimen'].astype(str).str.lower().str.strip()
        df['at'] = pd.to_datetime(df['at'], errors='coerce')

        return df
    return pd.DataFrame()

df = load_data()

# =========================
# SIDEBAR
# =========================
if os.path.exists(LOGO_PATH):
    st.sidebar.image(Image.open(LOGO_PATH), width=200)

st.sidebar.title("Filters")

if df.empty:
    st.error("❌ Dataset tidak ditemukan")
    st.stop()

# FILTER
min_date = df['at'].min().date()
max_date = df['at'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Tanggal",
    (min_date, max_date)
)

sentiments = st.sidebar.multiselect(
    "Sentimen",
    sorted(df['sentimen'].unique()),
    default=sorted(df['sentimen'].unique())
)

ratings = st.sidebar.multiselect(
    "Rating",
    sorted(df['score'].unique()),
    default=sorted(df['score'].unique())
)

df_filtered = df[
    (df['at'].dt.date >= start_date) &
    (df['at'].dt.date <= end_date) &
    (df['sentimen'].isin(sentiments)) &
    (df['score'].isin(ratings))
]

# =========================
# HEADER
# =========================
st.markdown(f"""
<div style='background-color:{DANA_BLUE};padding:15px;border-radius:10px'>
<h1 style='color:white;text-align:center;'>DANA App Monitoring Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# =========================
# KPI
# =========================
st.subheader("📊 KPI Overview")

total = len(df_filtered)
avg_rating = round(df_filtered['score'].mean(), 2)

sent_dist = df_filtered['sentimen'].value_counts(normalize=True) * 100
pos = round(sent_dist.get('positif', 0), 2)
neg = round(sent_dist.get('negatif', 0), 2)
net = round(sent_dist.get('netral', 0), 2)

critical = round(df_filtered['score'].isin([1,2]).mean()*100, 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Reviews", total)
c2.metric("Avg Rating", avg_rating)
c3.metric("Positive %", f"{pos}%")
c4.metric("Negative %", f"{neg}%")

c5, c6 = st.columns(2)
c5.metric("Neutral %", f"{net}%")
c6.metric("Critical Rate", f"{critical}%")

# =========================
# TREND
# =========================
st.subheader("📈 Review Trend")
trend = df_filtered.groupby(df_filtered['at'].dt.date).size()
st.line_chart(trend)

# =========================
# SENTIMENT VISUAL
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    fig = px.pie(df_filtered, names='sentimen', hole=0.5)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Sentiment vs Rating")
    pivot = pd.crosstab(df_filtered['score'], df_filtered['sentimen'], normalize='index')
    fig2 = px.bar(pivot, barmode='stack')
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# CRITICAL ISSUE (PAKAI KOLOM DATA)
# =========================
st.subheader("🚨 Critical Issues")

saldo = df_filtered['is_saldo_hilang'].sum()
premium = df_filtered['is_premium'].sum()

c1, c2 = st.columns(2)
c1.error(f"Saldo Hilang Cases: {saldo}")
c2.warning(f"Premium Issues: {premium}")

# =========================
# KEYWORDS
# =========================
st.subheader("🔤 Top Keywords")

text = " ".join(df_filtered['content'].dropna().str.lower())
words = text.split()

common = Counter(words).most_common(10)
df_words = pd.DataFrame(common, columns=['word', 'count'])

st.bar_chart(df_words.set_index('word'))

# =========================
# CHURN DETECTION (FIXED)
# =========================
st.subheader("⚠️ Churn Risk")

if 'userName' in df_filtered.columns:
    user_group = df_filtered.sort_values('at').groupby('userName')
    declining = 0

    for user, data in user_group:
        if len(data) >= 2:
            if data['score'].iloc[-1] <= data['score'].iloc[0] - 2:
                declining += 1

    st.warning(f"Declining Users: {declining}")
else:
    st.info("Kolom userName tidak ditemukan")

# =========================
# MISMATCH DETECTION
# =========================
st.subheader("⚠️ Sentiment vs Rating Mismatch")

mismatch = df_filtered[
    ((df_filtered['score'] <= 2) & (df_filtered['sentimen'] == 'positif')) |
    ((df_filtered['score'] >= 4) & (df_filtered['sentimen'] == 'negatif'))
]

st.write(f"Jumlah mismatch: {len(mismatch)}")

# =========================
# AUTO INSIGHT
# =========================
st.subheader("🧠 Insights")

if critical > 25:
    st.error("Critical rate tinggi! Banyak user tidak puas.")

if saldo > 10:
    st.error("Isu saldo hilang tinggi → trust issue!")

if premium > 20:
    st.warning("Fitur premium bermasalah.")

if pos < 60:
    st.warning("Sentimen positif masih rendah.")

# =========================
# DEBUG
# =========================
with st.expander("DEBUG"):
    st.write("Kolom:", df.columns)
    st.write("Files:", os.listdir())
