import pandas as pd
import joblib

# Load the best model, encoder, and scaler
model = joblib.load('best_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

categorical_features = ['severity', 'name', 'CWE_ID', 'found_by']

def make_prediction(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # Encode the categorical features
    encoded_features = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

    # Drop original categorical features and concatenate the encoded features
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, encoded_df], axis=1)

    # Standardize the age column
    df['age'] = scaler.transform(df[['age']])

    # Make a prediction
    prediction = model.predict(df)
    return 'Yes' if prediction[0] == 1 else 'No'

# Example usage
input_data = {
    'severity': 'Medium',
    'name': 'SQL Injection',
    'CWE_ID': 'CWE-89',
    'age': 25,
    'found_by': 'John Doe'
}

prediction = make_prediction(input_data)
print(f"Predicted Target: {prediction}")
