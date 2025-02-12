import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
data_path = r"D:\new researches\Car research2\features csv files\DB1_Categorized_features1.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(["label", "file_name"], axis=1)
y = df["label"]

# Apply label encoding to target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features for Neural Network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define base classifiers
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42)

# Train classifiers
mlp.fit(X_train, y_train)
extra_trees.fit(X_train, y_train)
lgbm.fit(X_train, y_train)

# Define weights for each classifier
weights = [0.6, 0.2, 0.2]  # Normalized weights (optional, but useful)

# Predict individual votes
pred1 = mlp.predict(X_test)
pred2 = extra_trees.predict(X_test)
pred3 = lgbm.predict(X_test)

# Combine predictions into an array
predictions = np.array([pred1, pred2, pred3])

# Apply weighted voting
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

# Compute final predictions (encoded labels)
weighted_votes_encoded = weighted_hard_voting(predictions, weights)

# Decode predictions back to original labels
weighted_votes_decoded = label_encoder.inverse_transform(weighted_votes_encoded)

# Evaluate performance
print(f"Weighted Hard Voting Accuracy: {accuracy_score(y_test, weighted_votes_encoded):.4f}")

# Print a sample predicted value
sample_index = 0  # Change index to check different samples
print(f"Sample Predicted Value: {weighted_votes_decoded[sample_index]}")

# ---- Example for one row prediction ----
# Select a row from X_test
sample_row = X_test[sample_index].reshape(1, -1)  # Reshaping to make it a 2D array

# Make individual predictions
mlp_pred = mlp.predict(sample_row)
extra_trees_pred = extra_trees.predict(sample_row)
lgbm_pred = lgbm.predict(sample_row)

# Decode predictions back to original labels for clarity
mlp_pred_label = label_encoder.inverse_transform(mlp_pred)
extra_trees_pred_label = label_encoder.inverse_transform(extra_trees_pred)
lgbm_pred_label = label_encoder.inverse_transform(lgbm_pred)

# Print predictions from each model
print("\nPredictions for sample row:")
print(f"MLP Prediction: {mlp_pred_label[0]}")
print(f"Extra Trees Prediction: {extra_trees_pred_label[0]}")
print(f"LightGBM Prediction: {lgbm_pred_label[0]}")
