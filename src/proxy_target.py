import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def create_proxy_target(df_txn:pd.DataFrame) -> pd.DataFrame:
    """
    Create a proxy target variable by clustering the feature set.

    Parameters:
    df_txn (pd.DataFrame): Original transaction dataframe.
    df_features (pd.DataFrame): Dataframe containing features for clustering.

    Returns:
    pd.DataFrame: Original dataframe with an added 'proxy_target' column.
    """
    # compute RFM
    snapshot_date = df_txn['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm_df = df_txn.groupby('CustomerId').agg(
        recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', 'sum')
        ).reset_index()

    # Scaling and applying KMeans clustering
    rfm_scaled = StandardScaler().fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Define high risk clusters (e.g., cluster with highest recency and lowest monetary)
    cluster_summary = rfm_df.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
    high_risk_cluster = cluster_summary.sort_values(by=['recency', 'frequency','monetary'], ascending=[False, True, True]).index[0]
    rfm_df['proxy_target'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)

    # Convert CustomerId to string for merging
    df_txn['CustomerId'] = df_txn['CustomerId'].astype(str)
    rfm_df['CustomerId'] = rfm_df['CustomerId'].astype(str)

    # Merge proxy target back to original dataframe
    df_txn = df_txn.merge(rfm_df[['CustomerId', 'proxy_target']], on='CustomerId', how='left')
    return df_txn