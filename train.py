# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load the generated data
data = pd.read_csv('Cybersecurity_Data.csv')

# Check the distribution of the target variable
print(data['target'].value_counts())

# Feature Engineering
# Convert categorical features to numerical using OneHotEncoder
categorical_features = ['severity', 'name', 'CWE_ID', 'found_by']
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_features])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Drop original categorical features and concatenate the encoded features
data = data.drop(columns=categorical_features + ['ID'])
data = pd.concat([data, encoded_df], axis=1)

# Standardize the age column
scaler = StandardScaler()
data['age'] = scaler.fit_transform(data[['age']])

# Prepare the data for training
X = data.drop(columns=['target'])
y = data['target'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert target to binary

# Resample the data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_rf_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"RandomForest Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Save the model, encoder, and scaler
joblib.dump(best_rf_model, 'best_rf_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Trying Gradient Boosting Classifier for comparison
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate the model
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"GradientBoosting Model Accuracy: {accuracy_gb:.2f}")
print(classification_report(y_test, y_pred_gb))

# Save the Gradient Boosting model if it's better
if accuracy_gb > accuracy:
    joblib.dump(gb_model, 'best_model.pkl')
    print("GradientBoosting model saved as best_model.pkl")
else:
    joblib.dump(best_rf_model, 'best_model.pkl')
    print("RandomForest model saved as best_model.pkl")
