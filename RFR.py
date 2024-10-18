import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

# Load the dataset
file_path = '/path_to_your_file/insurance.csv'
data = pd.read_csv(file_path)

# Step 1: Apply label encoding to categorical features
label_encoders = {}
categorical_columns = ['sex', 'smoker', 'region']

# Apply label encoding to each categorical column
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 2: Set 'charges' as target and normalize the remaining features
X = data.drop('charges', axis=1)
y = data['charges']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Hyperparameter tuning using RandomizedSearchCV for Random Forest
rf = RandomForestRegressor(random_state=42)

# Define hyperparameters to tune
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Randomized Search with cross-validation
random_search = RandomizedSearchCV(
    estimator=rf, param_distributions=param_dist,
    n_iter=50, scoring='neg_mean_squared_error',
    cv=5, verbose=2, random_state=42, n_jobs=-1
)

# Fit the model to the data
random_search.fit(X_train, y_train)

# Step 5: Evaluate the model
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Cross-validation score
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='r2')

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Average CV R²: {np.mean(cv_scores)}")

# Step 6: Dump the model using pickle
model_filename = 'tuned_random_forest_regressor.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_rf, file)

print(f"Model saved as {model_filename}")
