import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image
import datetime

# 1. Page Configuration
st.set_page_config(
    page_title='DANA App Sentiment & Health Monitoring',
    page_icon='📊',
    layout='wide'
)

# 2. Data Loading Function
@st.cache_data
def load_data():
    file_path = '/content/df_combined_final.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Convert 'at' column to datetime
        df['at'] = pd.to_datetime(df['at'])
        return df
    return pd.DataFrame()

df = load_data()

# 3. Sidebar Implementation
logo_path = '/content/logo dana.png'
if os.path.exists(logo_path):
    try:
        logo_img = Image.open(logo_path)
        st.sidebar.image(logo_img, width=200)
    except Exception as e:
        st.sidebar.error(f'Error loading logo: {e}')

st.sidebar.title("Dashboard Filters")

if not df.empty:
    # Date Filter
    min_date = df['at'].min().date()
    max_date = df['at'].max().date()
    
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Sentiment Filter
    sentiment_options = sorted(df['sentimen'].unique())
    selected_sentiments = st.sidebar.multiselect(
        'Select Sentiment Label',
        options=sentiment_options,
        default=sentiment_options
    )

    # Rating Filter
    rating_options = sorted(df['score'].unique())
    selected_ratings = st.sidebar.multiselect(
        'Select Star Rating',
        options=rating_options,
        default=rating_options
    )

    # 4. Applying Filters
    mask = (
        (df['at'].dt.date >= start_date) &
        (df['at'].dt.date <= end_date) &
        (df['sentimen'].isin(selected_sentiments)) &
        (df['score'].isin(selected_ratings))
    )
    df_filtered = df.loc[mask]

    st.sidebar.success(f"Filtered Data: {len(df_filtered)} reviews")
else:
    st.error("Dataset not found. Please ensure '/content/df_combined_final.csv' exists.")

import plotly.express as px

DANA_BLUE = '#1C85C7'

# 4. Dashboard Header
st.markdown(f"""
    <div style='background-color:{DANA_BLUE};padding:15px;border-radius:10px;margin-bottom:25px;'>
        <h1 style='color:white;text-align:center;margin:0;'>DANA App Sentiment & Health Monitoring</h1>
    </div>
    """, unsafe_allow_html=True)

# 5. KPI Scorecards
st.subheader('Key Performance Indicators (KPIs)')
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", "53,000")
col2.metric("Avg Rating", "3.74", delta="Target > 4.0", delta_color="inverse")
col3.metric("Positive Rate", "54.58%", delta="Target > 80%", delta_color="inverse")
col4.metric("Critical Rate (1-2 Stars)", "28.02%", delta="Target < 10%", delta_color="inverse")

# 6. Visualizations
st.markdown('---')
st.subheader('Sentiment Analysis Visualizations')
vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.write('**Sentiment Distribution (Donut Chart)**')
    sent_dist = pd.DataFrame({
        'Label': ['Positif', 'Negatif', 'Netral'],
        'Value': [54.58, 33.20, 12.22]
    })
    fig_donut = px.pie(
        sent_dist, values='Value', names='Label', hole=0.5,
        color='Label', color_discrete_map={'Positif': '#28A745', 'Negatif': '#DC3545', 'Netral': '#FFC107'}
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with vis_col2:
    st.write('**Sentiment Proportion per Rating (Stacked Bar)**')
    rating_prop = pd.DataFrame({
        'Rating': ['1', '2', '3', '4', '5'] * 3,
        'Sentimen': ['Negatif']*5 + ['Netral']*5 + ['Positif']*5,
        'Proportion': [0.74, 0.63, 0.42, 0.23, 0.15, 0.20, 0.25, 0.24, 0.11, 0.07, 0.06, 0.12, 0.34, 0.66, 0.78]
    })
    fig_stacked = px.bar(
        rating_prop, x='Rating', y='Proportion', color='Sentimen',
        color_discrete_map={'Positif': '#28A745', 'Negatif': '#DC3545', 'Netral': '#FFC107'},
        barmode='stack'
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

# 7. Keyword Analysis
st.subheader('Top 10 Keywords Comparison')
key_col1, key_col2 = st.columns(2)

with key_col1:
    st.write('*Positive Keywords*')
    pos_keywords = pd.DataFrame({
        'Keyword': ['sangat', 'bagus', 'mantap', 'bantu', 'dana', 'baik', 'mudah', 'aplikasi', 'transaksi', 'puas'],
        'Frequency': [7841, 5722, 3938, 3937, 3320, 2739, 2268, 1791, 1318, 1306]
    })
    fig_pos = px.bar(pos_keywords, x='Frequency', y='Keyword', orientation='h', color_discrete_sequence=['#28A745'])
    fig_pos.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_pos, use_container_width=True)

with key_col2:
    st.write('*Negative Keywords*')
    neg_keywords = pd.DataFrame({
        'Keyword': ['dana', 'nya', 'aplikasi', 'saldo', 'mau', 'uang', 'gak', 'ga', 'masuk', 'akun'],
        'Frequency': [9382, 3573, 3566, 3327, 2608, 2599, 2582, 2298, 2236, 2147]
    })
    fig_neg = px.bar(neg_keywords, x='Frequency', y='Keyword', orientation='h', color_discrete_sequence=['#DC3545'])
    fig_neg.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_neg, use_container_width=True)

# 8. Critical Issue Tracking & Churn Risk
st.markdown('---')
st.subheader('Critical Issues & Churn Monitoring')
issue_col1, issue_col2 = st.columns(2)

with issue_col1:
    st.info('**Technical Keyword Frequency (KPI 6 & 7)**')
    st.write("- 🚨 **'saldo hilang'**: 346 occurrences")
    st.write("- 🛠️ **'premium' complaints**: 1,724 occurrences")

with issue_col2:
    st.warning('**Predictive Churn Insights**')
    st.write("🎯 **High-Churn-Risk Users**: 296 identified")
    st.write("- persistent critics (multiple negatives only) and declining users identified.")

print('KPIs, charts, and issue tracking appended to app_v2.py.')
