from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the best model, encoder, and scaler
model = joblib.load('best_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

categorical_features = ['severity', 'name', 'CWE_ID', 'found_by']

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    df = pd.DataFrame([input_data])
    encoded_features = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, encoded_df], axis=1)
    df['age'] = scaler.transform(df[['age']])
    prediction = model.predict(df)
    prediction_label = 'Yes' if prediction[0] == 1 else 'No'
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
