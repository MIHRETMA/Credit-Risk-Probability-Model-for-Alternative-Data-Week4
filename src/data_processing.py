import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from optbinning import BinningProcess

from proxy_target import create_proxy_target

def aggregate_feautures(df: pd.DataFrame) ->pd.DataFrame:
    """
    create aggregates customer level

    """
    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_txn_amt = ("Amount", "sum"),
            avg_txn_amt = ("Amount", "mean"),
            total_count = ("TransactionId", "count"),
            std_txn_amt = ("Amount", "std")
        )
        .reset_index()
    )
    return df.merge(agg_df, on="CustomerId", how="left")


def feauture_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracting Hour, Day, Month, Year

    """
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["Txn_hour"] = df["TransactionStartTime"].dt.hour
    df["Txn_day"] = df["TransactionStartTime"].dt.day
    df["Txn_month"] = df["TransactionStartTime"].dt.month
    df["Txn_year"] = df["TransactionStartTime"].dt.year

    return df

def dataprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Build sklearn ColumnTransformer for preprocessing

    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler())
        ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )
    return preprocessor


def apply_WOE(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
   """
   Apply WoE using optbinning (numeric features only)
   """
   # Separate numeric and categorical
   numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
   if not numeric_cols:
       return X  # No numeric columns to process
   
   X_num = X[numeric_cols].copy()
   # --- WoE on numeric features ---
   binning_process = BinningProcess(
       variable_names=numeric_cols,
       max_n_bins=5
   )
   binning_process.fit(X_num, y)
   X_num_woe = binning_process.transform(X_num, metric="woe")

   X[numeric_cols] = X_num_woe
   return X

def feature_enginering_pipeline(df: pd.DataFrame, target_col: str="label", apply_woe_transform: bool=True):
    """
    complete feature enginering pipeline
    """
    # aggregate and extract features
    df = aggregate_feautures(df)
    df = feauture_extraction(df)

    # drop identifier columns
    df = df.drop(columns=["CustomerId", "TransactionId", "SubscriptionId", "AccountId", "BatchId",
                          "ProductId", "ProviderId", "CurrencyCode", "ProductCategory", "ChannelId"])

    # drop raw timestamp
    df = df.drop(columns=["TransactionStartTime"])

    # separate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # identify column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.to_list()

   # One-Hot encoding for low-cardinality
    if categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_ohe = pd.DataFrame(ohe.fit_transform(X[categorical_cols]),
                             columns=ohe.get_feature_names_out(categorical_cols),
                             index=X.index)
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, X_ohe], axis=1)
    
    # apply WOE
    if apply_woe_transform:
        X_woe = apply_WOE(X,y)

    # Standardize numeric features
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    if X.select_dtypes(include="object").shape[1] > 0:
        raise ValueError("There are still categorical features in the dataset after processing.")

    return X,y 

if __name__ == "__main__":
    df_txn = pd.read_csv('../data/raw/data.csv')
    df_txn['TransactionStartTime'] = pd.to_datetime(df_txn['TransactionStartTime'])

    # create proxy target and save
    df_with_target = create_proxy_target(df_txn)
    df_with_target.to_csv('../data/processed/X_feautures_with_target.csv', index=False)

    # run feature engineering pipeline
    X,y = feature_enginering_pipeline(df_with_target, target_col="proxy_target")

    # save processed data
    X.to_csv('../data/processed/X_features.csv', index=False)
    y.to_csv('../data/processed/y_target.csv', index=False)

    

    print("Feature engineering completed successfully.")


