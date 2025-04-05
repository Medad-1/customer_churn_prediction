# customer_churn_prediction

# 📊 Customer Churn Prediction Web App

## 🔍 Overview
This is a web-based machine learning application that predicts whether a customer is likely to churn. The goal is to help businesses take proactive measures to retain customers before they leave.

## 🛠️ Technologies Used
- **Flask (Python)**: Backend web framework.
- **Decision Tree Classifier**: ML model used for prediction.
- **Scikit-learn**: For preprocessing and training the model.
- **Tailwind CSS**: For designing a modern, responsive UI.
- **Jinja2**: Templating engine to render dynamic HTML pages.

## 💡 Why Customer Churn Prediction?
Customer churn is a major concern for industries such as telecom, SaaS, and banking. Predicting churn helps businesses:
- Retain customers
- Reduce acquisition costs
- Improve overall profitability

## 🧠 How It Works

### 1. User Input (Frontend)
**`index.html`** is a form where users input:
- Account length
- Area code
- Plan subscriptions (international/voice mail)
- Usage metrics (day, evening, night, intl minutes)
- Customer service calls
- Number of voicemail messages
- State

### 2. Backend Processing (Flask)
#### `/` Route:
- Loads the home page with the form.

#### `/predict` Route:
- Accepts POST request.
- Parses form data.
- Preprocesses the data:
  - Encodes categorical features (LabelEncoder, OneHotEncoder)
  - Scales numerical features (StandardScaler)
- Loads the pre-trained Decision Tree model.
- Makes a prediction.
- Renders the result page with the output.

### 3. Prediction Output (Result Page)
**`result.html`** displays:
- Whether the customer is likely to churn.
- Styled with Tailwind and uses a dynamic background.
- Conditional formatting: green for "No", red for "Yes".

## 🔍 Machine Learning Model
- **Model**: `DecisionTreeClassifier`
- **Preprocessing**:
  - Label Encoding (binary fields like plan types)
  - OneHotEncoding (state column)
  - Standard Scaling (numerical values)
- **Training**: On a dataset with customer usage and churn labels.

## 🎨 Features
- 🧠 Smart ML prediction.
- 🎨 Clean, modern UI with Tailwind.
- 🔁 Dynamic forms and feedback.
- ⚙️ Easily extendable for analytics.

## 🚀 Future Improvements
- Switch to more advanced models (Random Forest, XGBoost).
- Add data visualizations (charts, churn trends).
- Save prediction history.
- Enable user accounts and login sessions.

## 📂 Folder Structure
```
├── app.py              # Flask app
├── templates
│   ├── index.html      # Input form
│   └── result.html     # Prediction result display
├── static              # (optional) static assets
├── model.pkl           # Trained ML model
└── requirements.txt    # Dependencies
```

## ✅ How to Run
1. Clone the repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
python app.py
```
4. Open `http://localhost:5000` in your browser.

---

Built with ❤️ to help reduce churn and improve customer satisfaction!

