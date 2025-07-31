import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

import os


# Load models safely
if os.path.exists("kmeans_model.pkl") and os.path.exists("rfm_scaler.pkl"):
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
else:
    st.error("Model files not found. Please make sure 'kmeans_model.pkl' and 'rfm_scaler.pkl' are in the repo.")
    st.stop()

# Load models and data
@st.cache_data
def load_data():
    df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    return df

df = load_data()

# Load clustering model
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('rfm_scaler.pkl')

# Compute product similarity matrix
@st.cache_data
def build_product_similarity(df):
    pivot = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum').fillna(0)
    similarity = pd.DataFrame(cosine_similarity(pivot.T), 
                              index=pivot.columns, 
                              columns=pivot.columns)
    return similarity

similarity_df = build_product_similarity(df)

# App Title and Sidebar
st.set_page_config(page_title='Retail Analytics', layout='wide')
st.sidebar.title("📊 Dashboard Navigation")
page = st.sidebar.radio("Choose Page", ["Home", "Clustering", "Recommendation"])

# Home Page
if page == "Home":
    st.title("🏪 Online Retail Customer Insights")
    st.write("Use the sidebar to navigate between the **Product Recommendation** and **Customer Segmentation** modules.")

# Clustering Page
elif page == "Clustering":
    st.title("🧠 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1, value=180)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, step=1, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=10.0, value=200.0)

    if st.button("Predict Segment"):
        user_rfm = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        user_scaled = scaler.transform(user_rfm)
        cluster = kmeans.predict(user_scaled)[0]

        label_map = {
            0: 'Occasional Shopper',
            1: 'High-Value Customer',
            2: 'Regular Buyer',
            3: 'At-Risk Customer'
        }

        st.success(f"🧑‍💼 This customer belongs to: **{label_map.get(cluster, 'Unknown')}**")

# Recommendation Page
elif page == "Recommendation":
    st.title("🛍️ Product Recommender")

    product_input = st.text_input("Enter Product Name", "GREEN VINTAGE SPOT BEAKER")

    if st.button("Recommend"):
        if product_input not in similarity_df.columns:
            st.error("Product not found in database. Please check the spelling.")
        else:
            recommendations = similarity_df[product_input].sort_values(ascending=False)[1:6]
            st.subheader("Recommended Products:")
            for i, prod in enumerate(recommendations.index, 1):
                st.markdown(f"{i}. **{prod}**")

