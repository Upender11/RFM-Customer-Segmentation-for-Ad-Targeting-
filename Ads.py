import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------------
# 1. Data Loading and Preprocessing
# -----------------------------------------------------
# For this example, we'll use the Online Retail dataset.
# Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
try:
    df = pd.read_excel('/content/Online Retail.xlsx')
except FileNotFoundError:
    print("Dataset file 'Online Retail.xlsx' not found.")
    print("Please download it and place it in the correct directory.")
    exit()

print("--- Initial Data Overview ---")
print(df.head())

# --- Data Cleaning ---
# Remove rows with missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)
# Remove returns (Quantity < 0)
df = df[df['Quantity'] > 0]
# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)
# Calculate 'TotalPrice'
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


# --- Setup PDF for output ---
pdf_pages = PdfPages('customer_segmentation_report.pdf')
print("\nGenerating PDF report: customer_segmentation_report.pdf")


# -----------------------------------------------------
# 2. RFM (Recency, Frequency, Monetary) Feature Engineering
# -----------------------------------------------------
print("\n--- Calculating RFM Features ---")

# --- Recency ---
# Set a snapshot date for recency calculation (one day after the last transaction)
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
# Calculate the recency for each customer
recency_df = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
recency_df.rename(columns={'InvoiceDate': 'LastPurchaseDate'}, inplace=True)
recency_df['Recency'] = (snapshot_date - recency_df['LastPurchaseDate']).dt.days

# --- Frequency ---
frequency_df = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
frequency_df.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

# --- Monetary ---
monetary_df = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
monetary_df.rename(columns={'TotalPrice': 'Monetary'}, inplace=True)

# Merge RFM features into a single dataframe
rfm_df = pd.merge(recency_df, frequency_df, on='CustomerID')
rfm_df = pd.merge(rfm_df, monetary_df, on='CustomerID')
rfm_df.drop('LastPurchaseDate', axis=1, inplace=True)

print(rfm_df.head())


# -----------------------------------------------------
# 3. K-Means Clustering for Segmentation
# -----------------------------------------------------
print("\n--- Performing K-Means Clustering ---")

# Scale the RFM data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

# --- Use the Elbow Method to find the optimal number of clusters ---
wcss = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss[k] = kmeans.inertia_

# Plot the Elbow Method graph
fig = plt.figure(figsize=(10, 6))
plt.plot(list(wcss.keys()), list(wcss.values()), 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('The Elbow Method')
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

# From the elbow plot, let's choose k=4 as the optimal number of clusters
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, random_state=42)
kmeans.fit(rfm_scaled)

# Assign the cluster labels to the RFM dataframe
rfm_df['Cluster'] = kmeans.labels_


# -----------------------------------------------------
# 4. Cluster Visualization with PCA
# -----------------------------------------------------
print("\n--- Visualizing Clusters with PCA ---")

# Reduce RFM data to 2 dimensions for plotting
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm_df['PCA1'] = rfm_pca[:, 0]
rfm_df['PCA2'] = rfm_pca[:, 1]

# Plot the clusters
fig = plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=rfm_df, palette='viridis', s=50, alpha=0.7)
plt.title('Customer Segments Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Customer Segment')
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

# Analyze the characteristics of each cluster
cluster_analysis = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\n--- Average RFM Values for Each Cluster ---")
print(cluster_analysis)


# -----------------------------------------------------
# 5. Close PDF and Finish
# -----------------------------------------------------
pdf_pages.close()
print("\nPDF report 'customer_segmentation_report.pdf' has been successfully generated.")
