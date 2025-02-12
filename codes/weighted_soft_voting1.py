import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.special import softmax

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def weighted_soft_voting(prob_preds, weights):
    weights = np.array(weights)
    weighted_sum = np.zeros_like(prob_preds[0])
    
    for i, probs in enumerate(prob_preds):
        weighted_sum += weights[i] * probs
    
    activated = softmax(weighted_sum, axis=1)
    return np.argmax(activated, axis=1)

def evaluate_weighted_voting(data_path, weights, num_features):
    df = pd.read_csv(data_path)
    
    X = df.drop(["label", "file_name"], axis=1)
    y = df["label"]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_selected = selector.fit_transform(X, y_encoded)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=500)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_SEED)
    ada = AdaBoostClassifier(n_estimators=50, random_state=RANDOM_SEED)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        lr.fit(X_train, y_train)
        mlp.fit(X_train, y_train)
        ada.fit(X_train, y_train)
        
        prob1 = lr.predict_proba(X_test)
        prob2 = mlp.predict_proba(X_test)
        prob3 = ada.predict_proba(X_test)
        
        prob_preds = [prob1, prob2, prob3]
        y_pred = weighted_soft_voting(prob_preds, weights)
        
        scores.append(accuracy_score(y_test, y_pred))
    
    return np.mean(scores)

# Example usage:
if __name__ == "__main__":
    data_path = r"D:\new researches\CAR RESEARCH\features csv files\DB1_features1.csv"
    
    weights = [0.4, 0.3, 0.3]
    num_features = 33
    
    accuracy = evaluate_weighted_voting(data_path, weights, num_features)
    print(f"Mean Accuracy with Softmax Activation (10-fold CV): {accuracy:.4f}")