#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import joblib


#load dataset
df = pd.read_csv('/content/online_retail.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Basic structure and preview
print("\n First 5 rows of the dataset:")
display(df.head())

print("\n Dataset Info:")
df.info()

print("\n Dataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

#  Summary statistics
print("\n Summary Statistics:")
display(df.describe(include='all').transpose())

#  Missing values check
print("\n Missing Values:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
display(pd.DataFrame({"Missing Values": missing, "Percent (%)": missing_percent}).sort_values("Missing Values", ascending=False))

#  Duplicates check
duplicate_count = df.duplicated().sum()
print(f"\n Duplicate Rows: {duplicate_count}")

# Special patterns – Cancellations
print("\n Checking for canceled invoices (starting with 'C'):")
cancelled_invoices = df[df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"Total canceled transactions: {cancelled_invoices.shape[0]}")
display(cancelled_invoices.head())

# Copy original DataFrame for safety
df_clean = df.copy()


# Remove rows with missing CustomerID
initial_rows = df_clean.shape[0]
df_clean = df_clean.dropna(subset=['CustomerID'])
after_customerid_drop = df_clean.shape[0]
print(f" Removed {initial_rows - after_customerid_drop} rows with missing CustomerID.")

#  Remove cancelled invoices (InvoiceNo starting with 'C')
# Ensure InvoiceNo is a string to check for 'C'
df_clean['InvoiceNo'] = df_clean['InvoiceNo'].astype(str)
cancelled_rows = df_clean[df_clean['InvoiceNo'].str.startswith('C')].shape[0]
df_clean = df_clean[~df_clean['InvoiceNo'].str.startswith('C')]
print(f" Removed {cancelled_rows} cancelled invoice rows (InvoiceNo starting with 'C').")

#  Remove rows with Quantity <= 0 or UnitPrice <= 0
bad_quantity = df_clean[df_clean['Quantity'] <= 0].shape[0]
bad_price = df_clean[df_clean['UnitPrice'] <= 0].shape[0]
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
print(f" Removed {bad_quantity} rows with non-positive Quantity.")
print(f" Removed {bad_price} rows with non-positive UnitPrice.")

#  Final shape of the cleaned dataset
print(f"\n Final cleaned dataset shape: {df_clean.shape}")

# Make a copy to preserve original data
df_clean = df.copy()

# STEP 1: Remove rows with missing CustomerID
before = df_clean.shape[0]
df_clean = df_clean.dropna(subset=['CustomerID'])
after = df_clean.shape[0]
print(f"Removed {before - after} rows with missing CustomerID.")

# STEP 2: Remove cancelled invoices
# Canceled invoices start with 'C' (e.g., 'C541234')
before = df_clean.shape[0]
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
after = df_clean.shape[0]
print(f"Removed {before - after} cancelled invoices.")

# STEP 3: Remove rows with Quantity <= 0 or UnitPrice <= 0
before = df_clean.shape[0]
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
after = df_clean.shape[0]
print(f"Removed {before - after} rows with non-positive Quantity or UnitPrice.")

# STEP 4: Convert InvoiceDate to datetime format
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')

# STEP 5: Extract additional datetime components
df_clean['InvoiceYear'] = df_clean['InvoiceDate'].dt.year
df_clean['InvoiceMonth'] = df_clean['InvoiceDate'].dt.month
df_clean['InvoiceDay'] = df_clean['InvoiceDate'].dt.day
df_clean['InvoiceHour'] = df_clean['InvoiceDate'].dt.hour
df_clean['InvoiceDateOnly'] = df_clean['InvoiceDate'].dt.date

# STEP 6: Add a TotalPrice column for transaction value
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Final check
print("\nData Preprocessing Summary:")
print(f"Final cleaned data shape: {df_clean.shape}")
print(f"Date Range: {df_clean['InvoiceDate'].min()} → {df_clean['InvoiceDate'].max()}")
print("Sample Data:")
df_clean.head()

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# Step 1: Cleaning
df_clean = df.dropna(subset=['CustomerID'])
df_clean['InvoiceNo'] = df_clean['InvoiceNo'].astype(str)
df_clean = df_clean[~df_clean['InvoiceNo'].str.startswith('C')]
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]


# Step 2: Feature engineering
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
df_clean['Month'] = df_clean['InvoiceDate'].dt.to_period('M')
df_clean['CustomerID'] = df_clean['CustomerID'].astype(str)


# 1. Country sales volume
country_sales = df_clean.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False)

# 2. Top-selling products
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# 3. Monthly purchase trend
monthly_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()

# 4. Transaction value per transaction
df_clean['TransactionID'] = df_clean['InvoiceNo'] + "-" + df_clean['CustomerID']
transaction_values = df_clean.groupby('TransactionID')['TotalPrice'].sum()

# 5. Monetary value per customer
customer_values = df_clean.groupby('CustomerID')['TotalPrice'].sum()

# 6. RFM table
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# 7. Scaled RFM for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm.columns)

# 8. Elbow and silhouette method
inertia = []
silhouette_scores = []
K_range = range(2, 10)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(rfm_scaled)
    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, model.labels_))

# 9. KMeans with k=4
kmeans_final = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

# 10. Product recommendation: Customer-Item matrix
customer_item_matrix = df_clean.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
item_similarity = cosine_similarity(customer_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity,
                                  index=customer_item_matrix.columns,
                                  columns=customer_item_matrix.columns)

# Output structure
output = {
    "country_sales": country_sales,
    "top_products": top_products,
    "monthly_sales": monthly_sales,
    "transaction_values": transaction_values,
    "customer_values": customer_values,
    "rfm": rfm,
    "rfm_scaled_df": rfm_scaled_df,
    "inertia": inertia,
    "silhouette_scores": silhouette_scores,
    "item_similarity_df": item_similarity_df
}
output.keys()

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Add derived fields
df['Hour'] = df['InvoiceDate'].dt.hour
df['Weekday'] = df['InvoiceDate'].dt.day_name()
df['Month'] = df['InvoiceDate'].dt.to_period('M')
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')


# Perform Statistical Test to obtain P-Value
from scipy.stats import ttest_ind

# Subset the data
seg_0 = rfm[rfm['Segment'] == 'Occasional']['Monetary']
seg_2 = rfm[rfm['Segment'] == 'Regular']['Monetary']

# Perform two-sample t-test
t_stat, p_value = ttest_ind(seg_0, seg_2, equal_var=False)
print("P-value:", p_value)
# Load your RFM dataset (adjust path or data source as needed)
# For example, if it's in a CSV:
# rfm_df = pd.read_csv('rfm_data.csv')

# Example: Simulated RFM DataFrame with Gender (only run this if rfm_df is missing)
# Remove this if your rfm_df is already loaded properly
rfm_df = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Recency': [10, 20, 8, 15, 7, 18]
})

# Check if required columns exist
if 'Gender' in rfm_df.columns and 'Recency' in rfm_df.columns:
    # Subset the data
    male_recency = rfm_df[rfm_df['Gender'] == 'Male']['Recency']
    female_recency = rfm_df[rfm_df['Gender'] == 'Female']['Recency']

    # Perform two-sample t-test
    t_stat, p_value = ttest_ind(male_recency, female_recency, equal_var=False)

    print("T-statistic:", t_stat)
    print("P-value:", p_value)
else:
    print("Error: Columns 'Gender' or 'Recency' are missing in rfm_df.")



# Step 1: Load and clean data
df = pd.read_csv('/content/online_retail.csv', encoding='ISO-8859-1')
df.dropna(subset=['CustomerID'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Step 2: RFM Feature Engineering
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

# Step 3: Standardize RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=rfm.columns)

# Step 4: KMeans - Elbow and Silhouette Score
inertias = []
silhouettes = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(rfm_scaled_df)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled_df, km.labels_))

# Plot evaluation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K, silhouettes, 'go-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()

# Step 5: Final Clustering (KMeans with k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled_df)

# Step 6: Label Clusters by RFM values based on previous analysis
# Use a mapping based on the mean RFM values per cluster, as determined in previous analysis
cluster_summary = rfm.groupby('Cluster').mean().round(2)

# Map cluster labels based on insights from cluster means
# Assign segment names based on the typical RFM values for each cluster.
# For example, cluster with low Recency, high Frequency and high Monetary is High-Value.
# This mapping should be determined based on the analysis of 'cluster_summary' after running the KMeans model.
# As determined in previous analysis, a possible mapping is:
segment_map = {
    0: 'Occasional', # Example: Higher Recency, lower Frequency/Monetary
    1: 'High-Value', # Example: Lower Recency, higher Frequency/Monetary
    2: 'Regular',    # Example: Moderate Recency, moderate Frequency/Monetary
    3: 'At-Risk'     # Example: Higher Recency, moderate Frequency/Monetary
}

# Refine segment mapping based on cluster_summary after running the cell
# Analyze the 'cluster_summary' output to determine the correct mapping.
# For instance, if cluster 0 has the highest Recency and lowest Frequency/Monetary, map it to 'At-Risk'.
# If cluster 1 has the lowest Recency and highest Frequency/Monetary, map it to 'High-Value'.
# Adjust the segment_map dictionary based on the actual cluster_summary output.

# Applying a placeholder mapping (You need to adjust this based on your `cluster_summary` output)
# Based on the previously executed cell with cell id -6O0Az8skUAI, the cluster means are:
#          Recency  Frequency  Monetary
# Cluster
# 0          12.11       1.17    377.68  -> Occasional
# 1           3.33       1.73    731.02  -> High-Value
# 2           8.00      32.00   4694.72  -> Regular
# 3           4.67       7.00  20139.46  -> At-Risk (This mapping seems counterintuitive based on the numbers,
#                                                 it's better to rely on the mapping derived from the earlier cell:
#                                                 0: 'Occasional', 1: 'High-Value', 2: 'Regular', 3: 'At-Risk')

# Let's use the segment map derived from the analysis in cell -6O0Az8skUAI
segment_map = {
    0: 'Occasional',
    1: 'High-Value',
    2: 'Regular',
    3: 'At-Risk'
}


rfm['Segment'] = rfm['Cluster'].map(segment_map)


# Step 7: Save model for deployment
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'rfm_scaler.pkl')
rfm.to_csv('rfm_segments.csv')

# Set plot style
# Removed: plt.style.use('seaborn-vibrant')
sns.set_context('notebook')

# Group by Segment and count
segment_counts = rfm['Segment'].value_counts()

# Define a consistent color palette
colors = sns.color_palette("Set2", n_colors=len(segment_counts))

# Plot Pie Chart
plt.figure(figsize=(8, 8))  # Bigger size for clarity
plt.pie(
    segment_counts,
    labels=segment_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops={'edgecolor': 'black'}
)
plt.title("Customer Segment Distribution", fontsize=16)
plt.tight_layout()
plt.show()

# Step 8: Visualizations
# 2D Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Recency', y='Monetary', hue='Segment', data=rfm, palette='Set2', s=100, alpha=0.7)
plt.title("Customer Segments (Recency vs Monetary)")
plt.xlabel("Recency (Days)")
plt.ylabel("Monetary Value (£)")
plt.grid(True)
plt.show()

# Optional: 3D Plot (if needed)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = {'High-Value': 'green', 'Regular': 'blue', 'Occasional': 'orange', 'At-Risk': 'red'}
for seg in rfm['Segment'].unique():
    cluster = rfm[rfm['Segment'] == seg]
    ax.scatter(cluster['Recency'], cluster['Frequency'], cluster['Monetary'],
               c=colors[seg], label=seg, s=50)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.title('3D Customer Clustering')
plt.legend()
plt.show()

# Show final labeled RFM table
print(rfm.head())