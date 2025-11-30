import streamlit as st
import pandas as pd
import joblib
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
import plotly.express as px
import numpy as np

# -----------------------------
# Page Config & Title
# -----------------------------
st.set_page_config(page_title="Accident Risk Heatmap MVP", layout="wide")
st.title("🚧 Predicted Accident Hotspot Viewer (MVP)")

st.markdown("""
This MVP visualizes **predicted accident severity hotspots** in Thailand based on your uploaded accident CSV.  

- **Folium Heatmap:** Shows risk hotspots weighted by the model's predicted probability of high severity.  
- **Plotly Density Map:** Shows the same predictions with interactive zoom/pan.  

Each point’s brightness represents the model's confidence that the accident is high severity.  

**Upload CSV:** Must contain the following columns:  
`incident_datetime`, `latitude`, `longitude`, `number_of_vehicles_involved`, `vehicle_type`, `presumed_cause`, `accident_type`, `weather_condition`, `road_description`, `slope_description`
""")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded = st.file_uploader("Upload Accident CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    
    required_cols = ["incident_datetime",
        "latitude", "longitude", "number_of_vehicles_involved",
        "vehicle_type","presumed_cause","accident_type",
        "weather_condition","road_description","slope_description"
    ]
    
    if not set(required_cols).issubset(df.columns):
        st.error(f"CSV must contain the required columns: {required_cols}")
        st.stop()
    
    st.success(f"Loaded {len(df)} accident records.")
    
    # -----------------------------
    # Load XGBoost model
    # -----------------------------
    model = joblib.load("BEST_XGBOOST_MODEL.pkl")
    
    # -----------------------------
    # Preprocess & Predict
    # -----------------------------
    def preprocess_and_predict(df_input, model):
        df_pred = df_input.copy()
        df_pred['lat_round3'] = df_pred['latitude'].round(3)
        df_pred['lon_round3'] = df_pred['longitude'].round(3)

        # Count accidents per grid
        grid_counts = df_pred.groupby(['lat_round3','lon_round3']).size().rename('grid_count').reset_index()
        df_pred = df_pred.merge(grid_counts, on=['lat_round3','lon_round3'], how='left')
        df_pred['grid_count'] = df_pred['grid_count'].fillna(0).astype(int)

        # Datetime features
        df_pred['incident_datetime'] = pd.to_datetime(df_pred['incident_datetime'], errors='coerce')
        df_pred = df_pred.dropna(subset=['incident_datetime']).reset_index(drop=True)
        df_pred['hour'] = df_pred['incident_datetime'].dt.hour
        df_pred['weekday'] = df_pred['incident_datetime'].dt.weekday
        df_pred['month'] = df_pred['incident_datetime'].dt.month

        numeric_feats = ['number_of_vehicles_involved','latitude','longitude','grid_count','hour','weekday','month']
        cat_feats = ['province_en','vehicle_type','presumed_cause','accident_type','weather_condition','road_description','slope_description']

        # Fill missing categorical/numeric features
        for col in numeric_feats:
            if col not in df_pred.columns:
                df_pred[col] = 0
        for col in cat_feats:
            if col not in df_pred.columns:
                df_pred[col] = 'Missing'

        X_pred = df_pred[numeric_feats + cat_feats]
        y_proba = model.predict_proba(X_pred)[:,1]
        df_pred['severity_pred'] = np.where(y_proba>=0.5,'high','low')
        df_pred['severity_conf'] = y_proba

        return df_pred[['latitude','longitude','severity_pred','severity_conf']]

    df_pred = preprocess_and_predict(df, model)
    
    # -----------------------------
    # Folium Heatmap (weighted)
    # -----------------------------
    st.subheader("Folium Heatmap: Predicted Severity Risk")
    center_lat = df_pred["latitude"].median()
    center_lon = df_pred["longitude"].median()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    heat_data = df_pred[['latitude', 'longitude', 'severity_conf']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, max_val=1).add_to(m)
    
    st_folium(m, width=850, height=600)
    
    # -----------------------------
    # Plotly Density Heatmap
    # -----------------------------
    st.subheader("Plotly Density Map: Predicted Severity Risk")
    
    fig = px.density_mapbox(
        df_pred,
        lat="latitude",
        lon="longitude",
        z="severity_conf",        # weighted by model confidence
        radius=15,
        center=dict(lat=center_lat, lon=center_lon),
        zoom=5.8,
        mapbox_style="open-street-map",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # -----------------------------
    # Top predicted risk points
    # -----------------------------
    st.subheader("Top 10 High-Risk Predictions")
    top_risk = df_pred.sort_values('severity_conf', ascending=False).head(10)
    st.dataframe(top_risk)

else:
    st.info("Upload a CSV to continue.")
