import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, jsonify, request, render_template

# Load dataset
file_path = "churn-bigml-20.csv"
df = pd.read_csv(file_path)

# Label Encoding for binary categorical columns
df['International plan'] = LabelEncoder().fit_transform(df['International plan'])
df['Voice mail plan'] = LabelEncoder().fit_transform(df['Voice mail plan'])

# One-Hot Encoding for 'State'
df = pd.get_dummies(df, columns=['State'], drop_first=True)

# Features and target
X = df.drop(columns=['Churn'])
y = df['Churn'].astype(int)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Evaluation
y_pred = dt_classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Customer behavior dictionary
customer_behavior = {
    idx: {
        "Account Length": row["Account length"],
        "International Plan": row["International plan"],
        "Voice Mail Plan": row["Voice mail plan"],
        "Total Day Minutes": row["Total day minutes"],
        "Total Eve Minutes": row["Total eve minutes"],
        "Total Night Minutes": row["Total night minutes"],
        "Total Intl Minutes": row["Total intl minutes"],
        "Total Calls": row["Total day calls"] + row["Total eve calls"] + row["Total night calls"] + row["Total intl calls"],
        "Total Charges": row["Total day charge"] + row["Total eve charge"] + row["Total night charge"] + row["Total intl charge"],
        "Customer Service Calls": row["Customer service calls"],
        "Churn": bool(row["Churn"])
    }
    for idx, row in df.iterrows()
}

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', features=X.columns, prediction=None, error=None)

@app.route('/customer/<int:customer_id>', methods=['GET'])
def get_customer_behavior(customer_id):
    customer = customer_behavior.get(customer_id)
    if customer:
        return jsonify(customer)
    return jsonify({"error": "Customer not found"}), 404

@app.route('/predict', methods=['POST'])
def predict_churn():
    data = request.form
    try:
        # Initialize all features to zero
        new_data = {col: 0.0 for col in X.columns}

        # Critical inputs to validate separately
        required_fields = ["Number vmail messages"]
        missing_critical = []

        # Handle each feature
        for feature in new_data:
            if feature.startswith('State_'):
                continue  # Handle state separately

            value = data.get(feature)
            if value is None or value.strip() == "":
                # If it's a critical input, track it
                if feature in required_fields:
                    missing_critical.append(feature)
                # Otherwise, default to 0.0
                new_data[feature] = 0.0
            else:
                try:
                    new_data[feature] = float(value)
                except ValueError:
                    return render_template('index.html', features=X.columns, prediction=None, error=f"⚠️ Invalid number format for: {feature}")

        # Handle state input
        selected_state_col = data.get("State")  # e.g., "State_CA"
        if selected_state_col and selected_state_col in new_data:
            new_data[selected_state_col] = 1.0
        else:
            return render_template('index.html', features=X.columns, prediction=None, error="⚠️ Invalid or missing state selection.")

        # If any required fields are missing
        if missing_critical:
            return render_template('index.html', features=X.columns, prediction=None,
                                   error=f"⚠️ Missing required fields: {', '.join(missing_critical)}")

        # Convert to DataFrame and scale
        input_df = pd.DataFrame([new_data])
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = dt_classifier.predict(input_scaled)
        
        # Redirect to result page with prediction
        return render_template('result.html', prediction=bool(prediction[0]))

    except Exception as e:
        return render_template('index.html', features=X.columns, prediction=None, error=f"⚠️ {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)