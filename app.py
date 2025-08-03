import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------
# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df

df = load_data()

# -------------------------------------------
# Compute RFM and train model if missing
@st.cache_data
def compute_rfm_and_model(df):
    latest_date = df["InvoiceDate"].max()
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (latest_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)

    # Save model and scaler
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(scaler, "rfm_scaler.pkl")

    return rfm, kmeans, scaler

# Load or train models
if os.path.exists("kmeans_model.pkl") and os.path.exists("rfm_scaler.pkl"):
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    latest_date = df["InvoiceDate"].max()
else:
    rfm_data, kmeans, scaler = compute_rfm_and_model(df)

# -------------------------------------------
# Build product similarity matrix
@st.cache_data
def build_similarity_matrix(df):
    pivot = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum').fillna(0)
    similarity = pd.DataFrame(cosine_similarity(pivot.T), 
                              index=pivot.columns, 
                              columns=pivot.columns)
    return similarity

similarity_df = build_similarity_matrix(df)

# -------------------------------------------
# Streamlit UI
st.set_page_config(page_title="🛍 Shopper Spectrum", layout="wide")
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Choose Page", [" Home", " Clustering", " Recommendation"])

# -------------------------------------------
# Home Page
if page == " Home":
    st.title(" Shopper Spectrum: Retail Analytics")
    st.markdown("""
    This app provides:
    -  **Customer Segmentation** using K-Means on RFM metrics
    -  **Product Recommendations** using collaborative filtering
    
    Navigate using the sidebar to explore both modules.
    """)

# -------------------------------------------
# Clustering Page
elif page == " Clustering":
    st.title(" Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=100)
    frequency = st.number_input("Frequency (total purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=200.0)

    if st.button("Predict Segment"):
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        scaled_input = scaler.transform(input_df)
        cluster = kmeans.predict(scaled_input)[0]

        segment_map = {
            0: "Occasional Shopper",
            1: "High-Value Customer",
            2: "Regular Buyer",
            3: "At-Risk Customer"
        }

        st.success(f" Predicted Segment: **{segment_map.get(cluster, 'Unknown')}**")

# -------------------------------------------
# Recommendation Page
elif page == " Recommendation":
    st.title(" Product Recommender System")

    product_name = st.text_input("Enter Product Name:", "WHITE HANGING HEART T-LIGHT HOLDER")

    if st.button("Get Recommendations"):
        if product_name not in similarity_df.columns:
            st.error(" Product not found. Please check the spelling.")
        else:
            top_products = similarity_df[product_name].sort_values(ascending=False)[1:6]
            st.subheader(" Recommended Products:")
            for i, product in enumerate(top_products.index, start=1):
                st.markdown(f"{i}. **{product}**")


