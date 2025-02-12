import numpy as np 
from bayes_opt import BayesianOptimization
from skopt.space import Real  # Importing Real from scikit-optimize (skopt)
from weighted_soft_voting1 import evaluate_weighted_voting # Import the correct function
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define parameter bounds for Bayesian Optimization
weights_bounds = {
    'w1': (0, 1),
    'w2': (0, 1),
    'w3': (0, 1),
    'num_features': (20, 30) # Define range for num_features
}

# File path for dataset
data_path = r"D:\new researches\CAR RESEARCH\features csv files\DB2_features1.csv"

# Define an objective function wrapper for Bayesian Optimization
def objective(w1, w2, w3, num_features):
    weights = [w1, w2, w3]  # Ensure correct weight format
    return evaluate_weighted_voting(data_path, weights, int(num_features))  # Pass num_features as an integer

# Create Bayesian Optimization object
optimizer = BayesianOptimization(
    f=objective,
    pbounds=weights_bounds,
    random_state=42
)

# Perform optimization
optimizer.maximize(
    init_points=10,  # Number of initial random points
    n_iter=20       # Number of iterations
)

# Optimize Soft Voting
best_params = optimizer.max['params']  # Extract best parameters
best_weights = [best_params['w1'], best_params['w2'], best_params['w3']]
best_num_features = int(best_params['num_features'])
accuracy =evaluate_weighted_voting(data_path, best_weights,best_num_features)  # Pass best num_features

print(f"Best Weights: {best_weights}, Best Num Features: {best_num_features}, Accuracy: {accuracy}")
