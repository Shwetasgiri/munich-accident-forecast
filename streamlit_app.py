import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Munich Accident Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('accidents.csv')
    df = df[(df['MONAT'] != 'Summe') & (df['AUSPRAEGUNG'] == 'insgesamt')]
    df['date'] = pd.to_datetime(df['MONAT'], format='%Y%m')
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    return df

df_all = load_data()
model = joblib.load('model.pkl')
categories = ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle']

st.title("🚗 Munich Traffic Accident Analysis")

# Create Tabs to separate existing work from the new performance view
tab1, tab2 = st.tabs(["📈 Historical & Prediction", "🎯 Model Performance (2021)"])

with tab1:
    # --- HISTORICAL VISUALIZATION (UNCHANGED) ---
    st.header("Historical Trends & Yearly Comparison")
    sel_cat = st.selectbox("Select Category:", categories)
    hist_df = df_all[(df_all['MONATSZAHL'] == sel_cat) & (df_all['Year'] <= 2020)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Timeline (2000-2020)")
        fig1 = px.line(hist_df.sort_values('date'), x='date', y='WERT')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Yearly Side-by-Side (Seasonality)")
        seasonal_df = hist_df[hist_df['Year'] >= 2015]
        fig2 = px.line(seasonal_df, x='Month', y='WERT', color='Year')
        fig2.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig2, use_container_width=True)

    # --- PREDICTION SECTION ---
    st.divider()
    st.header("🔮 2021 Single Month Prediction")
    u_month = st.slider("Select Month for 2021", 1, 12, 1)

    if st.button("Predict & Calculate Error"):
        target_date = pd.to_datetime(f"2021-{u_month:02d}-01")
        forecast = model.get_prediction(start=target_date, end=target_date)
        pred_val = int(round(np.expm1(forecast.predicted_mean[0])))
        
        actual_row = df_all[(df_all['date'] == target_date) & (df_all['MONATSZAHL'] == 'Alkoholunfälle')]
        
        if not actual_row.empty:
            actual_val = int(actual_row['WERT'].values[0])
            abs_error = abs(pred_val - actual_val)
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted", pred_val)
            c2.metric("Actual (2021)", actual_val)
            c3.metric("Absolute Error", abs_error, delta=f"-{abs_error}", delta_color="inverse")

with tab2:
    st.header("Performance Analysis: Full Year 2021")
    st.write("Comparing the SARIMA model's forecast against the ground truth for 'Alkoholunfälle'.")

    # 1. Generate predictions for all 12 months of 2021
    months_2021 = pd.date_range(start='2021-01-01', end='2021-12-01', freq='MS')
    forecast_2021 = model.get_prediction(start=months_2021[0], end=months_2021[-1])
    preds = np.expm1(forecast_2021.predicted_mean).round()

    # 2. Get actuals for 2021
    actuals_df = df_all[(df_all['Year'] == 2021) & (df_all['MONATSZAHL'] == 'Alkoholunfälle')].sort_values('date')
    
    # 3. Create Comparison Table
    comparison_df = pd.DataFrame({
        'Month': actuals_df['date'],
        'Actual': actuals_df['WERT'].values,
        'Predicted': preds.values
    })

    # 4. Plot Comparison
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=comparison_df['Month'], y=comparison_df['Actual'], name='Actual Values', line=dict(color='blue', width=3)))
    fig_perf.add_trace(go.Scatter(x=comparison_df['Month'], y=comparison_df['Predicted'], name='Model Prediction', line=dict(color='orange', dash='dash')))
    
    fig_perf.update_layout(title="Actual vs Predicted Accidents (2021)", xaxis_title="Date", yaxis_title="Number of Accidents")
    st.plotly_chart(fig_perf, use_container_width=True)

    # 5. Show Metrics
    mae = np.mean(np.abs(comparison_df['Actual'] - comparison_df['Predicted']))
    st.metric("Average MAE for 2021", f"{mae:.2f} accidents")