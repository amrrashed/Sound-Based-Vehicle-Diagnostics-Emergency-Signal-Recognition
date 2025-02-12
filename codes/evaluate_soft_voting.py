import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

def evaluate_weighted_voting(data_path, weights, num_features):
    """
    Evaluates weighted hard voting ensemble with Logistic Regression, Neural Network, and SGD using 10-fold cross-validation.
    
    Parameters:
        data_path (str): Path to the dataset CSV file.
        weights (list): List of weights for the classifiers.
        num_features (int): Number of top features to select using ANOVA F-score.
    
    Returns:
        float: Mean accuracy of the weighted hard voting ensemble across 10 folds.
    """
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(["label", "file_name"], axis=1)
    y = df["label"]
    
    # Apply label encoding to target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Feature selection using ANOVA F-score
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_selected = selector.fit_transform(X, y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Define base classifiers
    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=500)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_SEED)
    sgd = SGDClassifier(loss='log_loss', random_state=RANDOM_SEED)
    
    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        # Train classifiers
        lr.fit(X_train, y_train)
        mlp.fit(X_train, y_train)
        sgd.fit(X_train, y_train)
        
        # Predict individual votes
        pred1 = lr.predict(X_test)
        pred2 = mlp.predict(X_test)
        pred3 = sgd.predict(X_test)
        
        # Combine predictions into an array
        predictions = np.array([pred1, pred2, pred3])
        
        # Apply weighted voting
        weighted_votes_encoded = weighted_hard_voting(predictions, weights)
        
        # Compute accuracy
        scores.append(accuracy_score(y_test, weighted_votes_encoded))
    
    return np.mean(scores)

# Example usage:
# data_path = "path/to/your/dataset.csv"
# weights = [0.4, 0.3, 0.3]
# num_features = 20
# accuracy = evaluate_weighted_voting(data_path, weights, num_features)
# print(f"Mean Accuracy with 10-fold CV: {accuracy:.4f}")
