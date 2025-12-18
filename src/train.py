import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score,
   f1_score, roc_auc_score
)
from data_processing import feature_enginering_pipeline

RANDOM_STATE = 42

def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluate classification model performance
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def train_log_model(model, param_grid, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate logistic regression model with hyperparameter tuning
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)

    # Log params and metrics to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics(metrics)

    # Log the model
    mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name=model_name)

    return metrics["roc_auc"]

if __name__ == "__main__":
    mlflow.set_experiment("Credit Risk Model Training")
    df = pd.read_csv('../data/processed/X_feautures_with_target.csv')
    X, y = feature_enginering_pipeline(df, target_col="proxy_target", apply_woe_transform=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Logistic Regression
    log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    log_reg_param = {'C': [0.01, 0.1, 1, 10]}
    lr_auc = train_log_model(
        log_reg, log_reg_param,
        X_train, X_test, y_train, y_test,
        model_name="Logistic_Regression_Credit_Risk_Model")
    
    print(f"Logistic Regression ROC AUC: {lr_auc}")

    # Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_param = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    rf_auc = train_log_model(
        rf, rf_param,
        X_train, X_test, y_train, y_test,
        model_name="Random_Forest_Credit_Risk_Model")
    
    print(f"Random Forest ROC AUC: {rf_auc}")

    print("Model training and evaluation completed.")