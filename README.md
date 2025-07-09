# RFM Customer Segmentation for Ad Targeting

## Description

This project segments online retail customers using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering. It visualizes these segments with PCA, enabling the creation of targeted ad campaigns for distinct groups like loyal, new, and dormant customers.

## Tech Stack

-   **Data Extraction & Preparation:** SQL
-   **Data Analysis & Feature Engineering:** Python (Pandas, NumPy, Datetime)
-   **Machine Learning Model:** Python (Scikit-learn)
    -   *Clustering Algorithm:* K-Means
    -   *Dimensionality Reduction:* PCA (Principal Component Analysis) for visualization
-   **Data Visualization:** Matplotlib, Seaborn

## Project Workflow

1.  **Data Extraction (SQL):**
    -   The first step is to query the database to get raw transaction data. This typically includes `customer_id`, `invoice_date`, and `unit_price`/`quantity`.

    ```sql
    -- Fetch all customer transaction data
    SELECT
        CustomerID,
        InvoiceNo,
        InvoiceDate,
        Quantity,
        UnitPrice
    FROM
        OnlineRetail
    WHERE
        CustomerID IS NOT NULL AND Quantity > 0 AND UnitPrice > 0;
    ```

2.  **RFM Feature Engineering (Python):**
    -   Using the transaction data, we calculate RFM (Recency, Frequency, Monetary) scores for each customer:
        -   **Recency:** How many days have passed since the customer's last purchase? (Lower is better)
        -   **Frequency:** How many distinct purchases has the customer made? (Higher is better)
        -   **Monetary:** What is the total amount of money the customer has spent? (Higher is better)

3.  **Customer Segmentation with K-Means:**
    -   The RFM features are first scaled to ensure they have an equal impact on the clustering algorithm.
    -   The K-Means clustering algorithm is then applied to group customers into a predefined number of segments (clusters). The optimal number of clusters is determined using the "Elbow Method."

4.  **Cluster Visualization with PCA:**
    -   Since it's difficult to visualize data with more than two dimensions (like our three RFM features), we use Principal Component Analysis (PCA) to reduce the data to two principal components.
    -   These components are then plotted on a 2D scatter plot, with each point colored according to its assigned segment. This provides a clear visual representation of the distinct customer groups.

## Outcome for Marketing

The final output is a set of clearly defined customer segments that the marketing team can use to design separate campaigns:
-   **High-Value / Champions:** High frequency, high monetary value, recent purchasers. Target with loyalty programs and new product previews.
-   **At-Risk / Needs Attention:** High spenders who haven't purchased in a while. Target with personalized re-engagement campaigns.
-   **New Customers:** Recent buyers with low frequency. Target with welcome offers and onboarding materials to encourage repeat purchases.
-   **Dormant / Lost:** Low recency, frequency, and monetary value. Can be targeted with "we miss you" campaigns or excluded from general marketing to save costs.

## How to Use This Project

1.  **Data:** Place the required dataset (e.g., `Online Retail.xlsx`) in the same directory.
2.  **Dependencies:** Install necessary libraries: `pip install pandas scikit-learn matplotlib seaborn openpyxl`.
3.  **Run Script:** Execute the Python script (`customer_segmentation.py`). It will perform the analysis and generate a PDF report named `customer_segmentation_report.pdf` containing all the visualizations.
