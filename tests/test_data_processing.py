import pandas as pd
from src.data_processing import aggregate_feautures, feauture_extraction
def test_aggregate_features_columns():
   df = pd.DataFrame({
       "CustomerId": ["A", "A", "B"],
       "Amount": [100, 200, 50],
       "TransactionId": [1, 2, 3]
   })
   df_out = aggregate_feautures(df)
   assert "total_txn_amt" in df_out.columns
   assert "avg_txn_amt" in df_out.columns
   assert "total_count" in df_out.columns

def test_feature_extraction_columns():
   df = pd.DataFrame({
       "TransactionStartTime": ["2024-01-01 10:00:00"]
   })
   df_out = feauture_extraction(df)
   assert "Txn_hour" in df_out.columns
   assert "Txn_day" in df_out.columns
   assert "Txn_month" in df_out.columns
   assert "Txn_year" in df_out.columns