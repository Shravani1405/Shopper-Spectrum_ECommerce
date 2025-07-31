#  Retail Analytics Web App (Streamlit)

This is a powerful **Streamlit web application** for:
1. **Customer Segmentation** using RFM (Recency, Frequency, Monetary) Clustering
2. **Product Recommendation** using Item-based Collaborative Filtering

---

### Features

###  Customer Segmentation (Clustering)

- Uses RFM (Recency, Frequency, Monetary) values
- Clusters customers into:
  - **High-Value**
  - **Regular**
  - **Occasional**
  - **At-Risk**
- Built using **KMeans** algorithm
- Input: Recency, Frequency, Monetary values
- Output: Cluster label (e.g., "High-Value Customer")

### Product Recommendation System

- Based on **item-based collaborative filtering**
- Uses **Cosine Similarity** between product purchases
- Input: Product Name
- Output: 5 most similar products

---

## Machine Learning Models Used

- `KMeans` for clustering customers
- `StandardScaler` for normalization
- `Cosine Similarity` for product-to-product recommendations
- Models are saved and loaded via `joblib`

---

##  Folder Structure

/project-root/
│
├── app.py # Main Streamlit App
├── online_retail.csv # Dataset
├── kmeans_model.pkl # Trained clustering model
├── rfm_scaler.pkl # Fitted scaler
├── rfm_segments.csv # Clustered customer data (optional)
├── requirements.txt # Python dependencies
└── README.md # Project documentation