import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score

#example
# data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
# weights = [0.6, 0.2, 0.2]  # Example weights
# accuracy = evaluate_weighted_voting(data_path, weights)
# print(f"Weighted Hard Voting Accuracy: {accuracy:.4f}")

def weighted_hard_voting(preds, weights):
    """
    Perform weighted hard voting.
    preds: array of shape (n_classifiers, n_samples)
    weights: list of weights corresponding to each classifier
    """
    unique_labels = np.unique(preds)  # Get unique class labels
    weighted_votes = np.zeros((preds.shape[1], len(unique_labels)))

    for i, label in enumerate(unique_labels):
        weighted_votes[:, i] = np.sum((preds == label) * np.array(weights)[:, None], axis=0)

    return unique_labels[np.argmax(weighted_votes, axis=1)]

def evaluate_weighted_voting(data_path, weights):
    # Set random seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(["label", "file_name"], axis=1)
    y = df["label"]
    
    # Apply label encoding to target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=RANDOM_SEED)
    
    # Define base classifiers
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_SEED)
    extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED)
    lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED)
    
    # Train classifiers
    mlp.fit(X_train, y_train)
    extra_trees.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    
    # Predict individual votes
    pred1 = mlp.predict(X_test)
    pred2 = extra_trees.predict(X_test)
    pred3 = lgbm.predict(X_test)
    
    # Combine predictions into an array
    predictions = np.array([pred1, pred2, pred3])
    
    # Apply weighted voting
    weighted_votes_encoded = weighted_hard_voting(predictions, weights)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, weighted_votes_encoded)
    return accuracy