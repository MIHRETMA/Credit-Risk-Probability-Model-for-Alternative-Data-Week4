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
   categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
   X_num = X[numeric_cols].copy()
   X_cat = X[categorical_cols].copy()
   # --- WoE on numeric features ---
   binning_process = BinningProcess(
       variable_names=numeric_cols,
       max_n_bins=5
   )
   binning_process.fit(X_num, y)
   X_num_woe = binning_process.transform(X_num, metric="woe")
   # --- Simple encoding for categorical (frequency encoding) ---
   for col in categorical_cols:
       freq_map = X_cat[col].value_counts(normalize=True)
       X_cat[col] = X_cat[col].map(freq_map)
   # Combine
   X_final = pd.concat([X_num_woe, X_cat], axis=1)
   return X_final

def feature_enginering_pipeline(df: pd.DataFrame, target_col: str="label", apply_woe_transform: bool=True):
    """
    complete feature enginering pipeline
    """
    # aggregate and extract features
    df = aggregate_feautures(df)
    df = feauture_extraction(df)

    # drop raw timestamp
    df = df.drop(columns=["TransactionStartTime"])

    # separate target
    y = df["FraudResult"]
    X = df.drop(columns=["FraudResult"])

    # identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
    categorical_cols = X.select_dtypes(include=["object"]).columns.to_list()

    # apply WOE
    if apply_woe_transform:
        X_woe = apply_WOE(X,y)
        return X_woe, y

    # build preprocessing pipeline
    preprocessor = dataprocessing_pipeline(numeric_cols, categorical_cols)

    # transform data
    X_processed = preprocessor.fit_transform(X)

    # convert back to a dataframe
    feature_names = (
        numeric_cols +
        list(
            preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(categorical_cols)
        )
    )

    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    

    return X_processed,y 

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/data.csv')
    df_txn = df.copy()
    df_txn['TransactionStartTime'] = pd.to_datetime(df_txn['TransactionStartTime'])

    # run feature engineering pipeline
    X,y = feature_enginering_pipeline(df)

    # save processed data
    X.to_csv('../data/processed/X_features.csv', index=False)
    y.to_csv('../data/processed/y_target.csv', index=False)

    # create proxy target and save
    df_processed = create_proxy_target(df_txn, X)
    df_processed.to_csv('../data/processed/X_feautures_with_target.csv', index=False)

    print("Feature engineering completed successfully.")


