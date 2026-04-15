import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image

# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(
    page_title='DANA App Sentiment & Health Monitoring',
    page_icon='📊',
    layout='wide'
)

DANA_BLUE = '#1C85C7'

# =========================
# 2. PATH SETUP (FIX UTAMA)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'df_combined_final.csv')
LOGO_PATH = os.path.join(BASE_DIR, 'logo dana.png')

# =========================
# 3. LOAD DATA
# =========================
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        # Safety check kolom
        if 'at' in df.columns:
            df['at'] = pd.to_datetime(df['at'], errors='coerce')

        return df
    return pd.DataFrame()

df = load_data()

# =========================
# 4. SIDEBAR
# =========================
# Logo
if os.path.exists(LOGO_PATH):
    try:
        logo = Image.open(LOGO_PATH)
        st.sidebar.image(logo, width=200)
    except:
        st.sidebar.warning("Logo gagal ditampilkan")

st.sidebar.title("Dashboard Filters")

if df.empty:
    st.error("❌ Dataset tidak ditemukan. Pastikan file CSV ada di folder project.")
    st.stop()

# =========================
# 5. FILTER
# =========================
min_date = df['at'].min().date()
max_date = df['at'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

sentiment_options = sorted(df['sentimen'].dropna().unique())
selected_sentiments = st.sidebar.multiselect(
    'Select Sentiment Label',
    options=sentiment_options,
    default=sentiment_options
)

rating_options = sorted(df['score'].dropna().unique())
selected_ratings = st.sidebar.multiselect(
    'Select Star Rating',
    options=rating_options,
    default=rating_options
)

# Apply filter
df_filtered = df[
    (df['at'].dt.date >= start_date) &
    (df['at'].dt.date <= end_date) &
    (df['sentimen'].isin(selected_sentiments)) &
    (df['score'].isin(selected_ratings))
]

st.sidebar.success(f"Filtered Data: {len(df_filtered)} reviews")

# =========================
# 6. HEADER
# =========================
st.markdown(f"""
<div style='background-color:{DANA_BLUE};padding:15px;border-radius:10px;margin-bottom:25px;'>
    <h1 style='color:white;text-align:center;'>DANA App Sentiment & Health Monitoring</h1>
</div>
""", unsafe_allow_html=True)

# =========================
# 7. KPI (DINAMIS)
# =========================
st.subheader('Key Performance Indicators (KPIs)')

total_reviews = len(df_filtered)
avg_rating = round(df_filtered['score'].mean(), 2)

positive_rate = round(
    (df_filtered['sentimen'] == 'Positif').mean() * 100, 2
)

critical_rate = round(
    (df_filtered['score'].isin([1, 2])).mean() * 100, 2
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", total_reviews)
col2.metric("Avg Rating", avg_rating)
col3.metric("Positive Rate", f"{positive_rate}%")
col4.metric("Critical Rate", f"{critical_rate}%")

# =========================
# 8. VISUALIZATION
# =========================
st.markdown('---')
st.subheader('Sentiment Analysis')

vis_col1, vis_col2 = st.columns(2)

# Donut Chart
with vis_col1:
    sent_count = df_filtered['sentimen'].value_counts().reset_index()
    sent_count.columns = ['Label', 'Value']

    fig_donut = px.pie(
        sent_count,
        values='Value',
        names='Label',
        hole=0.5
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# Stacked Bar
with vis_col2:
    rating_sent = df_filtered.groupby(['score', 'sentimen']).size().reset_index(name='count')

    fig_bar = px.bar(
        rating_sent,
        x='score',
        y='count',
        color='sentimen',
        barmode='stack'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# 9. KEYWORD (STATIC DEMO)
# =========================
st.subheader('Top Keywords')

col1, col2 = st.columns(2)

with col1:
    st.write("Positive Keywords")
    pos = pd.DataFrame({
        'Keyword': ['bagus', 'mantap', 'mudah'],
        'Frequency': [100, 80, 60]
    })
    st.bar_chart(pos.set_index('Keyword'))

with col2:
    st.write("Negative Keywords")
    neg = pd.DataFrame({
        'Keyword': ['error', 'gagal', 'saldo'],
        'Frequency': [120, 90, 70]
    })
    st.bar_chart(neg.set_index('Keyword'))

# =========================
# 10. DEBUG OPSIONAL
# =========================
with st.expander("🔍 Debug Info"):
    st.write("Current Directory:", os.getcwd())
    st.write("Files:", os.listdir())
