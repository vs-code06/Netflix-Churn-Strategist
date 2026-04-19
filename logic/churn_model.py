import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os

def train_model():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(ROOT_DIR, "data", "netflix_customer_churn.csv")
    df = pd.read_csv(CSV_PATH)

    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)

    X = df.drop("churned", axis=1)
    y = df["churned"]

    X = pd.get_dummies(X)
    X_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=10, 
        min_samples_leaf=2, 
        min_samples_split=10, 
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": cm.tolist()
    }

    return model, metrics, X_columns

def predict_new_customer(model, X_columns, customer_data):
    df_new = pd.DataFrame([customer_data])
    df_new = pd.get_dummies(df_new)
    df_new = df_new.reindex(columns=X_columns, fill_value=0)
    
    prediction = model.predict(df_new)
    return prediction[0]

def get_prediction_drivers(model, X_columns, customer_data, top_n=3):
    """
    Calculates which features contributed most to the 'Churn' prediction 
    for a specific user by tracing the Decision Path and measuring 
    the probability deltas at each split.
    """
    df_sample = pd.DataFrame([customer_data])
    df_sample = pd.get_dummies(df_sample)
    df_sample = df_sample.reindex(columns=X_columns, fill_value=0)
    
    # 1. Trace the path taken by this specific sample
    indicator = model.decision_path(df_sample)
    node_index = indicator.indices[indicator.indptr[0]:indicator.indptr[1]]
    
    # 2. Get the probability of 'Churn' (class 1) at each node in the path
    # model.tree_.value[node] = [[count_0, count_1]]
    path_values = model.tree_.value[node_index]
    node_sums = path_values.sum(axis=2).flatten()
    churn_probs = path_values[:, 0, 1] / node_sums
    
    # 3. Calculate deltas (how much each choice increased the risk)
    # delta[i] is the change caused by the feature at node_index[i]
    deltas = churn_probs[1:] - churn_probs[:-1]
    
    # Features used at each parent node in the path
    feature_indices = model.tree_.feature[node_index[:-1]]
    feature_names = [X_columns[f] for f in feature_indices]
    
    # 4. Aggregate deltas per feature (in case a feature is used multiple times)
    drivers = {}
    for name, delta in zip(feature_names, deltas):
        drivers[name] = drivers.get(name, 0.0) + delta
    
    # 5. Return top positive drivers (factors that increased churn risk)
    sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
    
    # Return as dict, filtering only factors that had a positive contribution to churn
    top_drivers = sorted_drivers[:top_n]
    return {k: round(float(v), 3) for k, v in top_drivers if v > 0}
