import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Munich Accident Forecast", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('accidents.csv')
    df = df[df['MONAT'] != 'Summe']
    df['date'] = pd.to_datetime(df['MONAT'], format='%Y%m')
    return df

df_all = load_data()
models = joblib.load('models_dict.pkl')
categories = ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle']
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Sidebar Controls
st.sidebar.header("🕹️ Controls")
selected_cat = st.sidebar.selectbox("Select Category", categories)
user_year = st.sidebar.number_input("Prediction Year", 2021, 2025, 2021)
user_month = st.sidebar.slider("Prediction Month", 1, 12, 1)

st.title(f"Traffic Accident Analysis: {selected_cat}")

# --- SECTION 1: HISTORICAL & SEASONAL ---
# Filter data strictly <= 2020 for the top charts
cat_hist = df_all[(df_all['MONATSZAHL'] == selected_cat) & 
                  (df_all['AUSPRAEGUNG'] == 'insgesamt') & 
                  (df_all['date'].dt.year <= 2020)].sort_values('date')

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Historical Trend (2000-2020)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=cat_hist['date'], y=cat_hist['WERT'], line=dict(color='#1f77b4')))
    fig_hist.update_layout(xaxis_title="Year", yaxis_title="Accident Count", template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("### 🗓️ Seasonal Averages")
    cat_hist['m'] = cat_hist['date'].dt.month
    seasonal = cat_hist.groupby('m')['WERT'].mean()
    fig_bar = go.Figure(go.Bar(x=month_names, y=seasonal, marker_color='#636efa'))
    fig_bar.update_layout(xaxis_title="Month", yaxis_title="Avg Accidents", template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- SECTION 2: PREDICTION ---
st.divider()
if st.sidebar.button("🚀 Generate Prediction"):
    current_model = models[selected_cat]
    target_date = pd.to_datetime(f"{user_year}-{user_month:02d}-01")
    
    # AI Forecast
    forecast = current_model.get_prediction(start=target_date, end=target_date)
    pred_val = int(forecast.predicted_mean[0])
    
    # Results Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("AI Prediction", pred_val)
    
    real_row = df_all[(df_all['date'] == target_date) & 
                       (df_all['MONATSZAHL'] == selected_cat) & 
                       (df_all['AUSPRAEGUNG'] == 'insgesamt')]
    
    if not real_row.empty:
        actual = int(real_row['WERT'].values[0])
        error = abs(pred_val - actual)
        c2.metric("Actual Reality", actual)
        c3.metric("Error (MAE)", error, delta=f"{((error/actual)*100):.1f}%", delta_color="inverse")
    else:
        c2.info("No ground truth data available for this date.")

    # Zoomed Forecast Chart
    st.markdown(f"### 🔭 Forecast vs. Reality: {selected_cat}")
    recent_df = df_all[(df_all['MONATSZAHL'] == selected_cat) & 
                       (df_all['AUSPRAEGUNG'] == 'insgesamt') & 
                       (df_all['date'] >= '2018-01-01')].sort_values('date')
    
    fig_pred = go.Figure()
    # Historical Blue Line
    fig_pred.add_trace(go.Scatter(
        x=recent_df[recent_df['date'] <= '2020-12-01']['date'], 
        y=recent_df[recent_df['date'] <= '2020-12-01']['WERT'], 
        name="Historical", line=dict(color="#2E86C1", width=3)
    ))
    # Actual Green Line (2021+)
    fig_pred.add_trace(go.Scatter(
        x=recent_df[recent_df['date'] >= '2021-01-01']['date'], 
        y=recent_df[recent_df['date'] >= '2021-01-01']['WERT'], 
        name="Actual Data", line=dict(color="#27AE60", dash='dot')
    ))
    # Prediction Point
    fig_pred.add_trace(go.Scatter(
        x=[target_date], y=[pred_val], mode="markers+text", 
        name="Prediction", text=[f"Pred: {pred_val}"], textposition="top center",
        marker=dict(color="crimson", size=18, symbol="star")
    ))
    
    fig_pred.update_layout(xaxis_title="Timeline", yaxis_title="Accidents", template="plotly_white")
    st.plotly_chart(fig_pred, use_container_width=True)