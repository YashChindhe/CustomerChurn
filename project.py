import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Customer_Churn.csv")
    
    # Mapping categorical to numeric
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})
    data['MultipleLines'] = data['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 2})
    data['InternetService'] = data['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    data['Contract'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    
    features = ['gender', 'SeniorCitizen', 'tenure', 'PhoneService', 'MultipleLines',
                'InternetService', 'Contract', 'MonthlyCharges']
    X = data[features].astype('float64')
    y = data['Churn']
    
    return X, y

X, y = load_data()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train models
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

svm_model = svm.SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Streamlit app
st.title("📊 Customer Churn Prediction")

st.markdown("Use the options in the sidebar to simulate customer behavior and see if they’re likely to churn.")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    senior = st.sidebar.selectbox('Senior Citizen', ['Yes', 'No'])
    tenure = st.sidebar.slider('Tenure (months)', 0, 75, 1)
    phone = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    multi_lines = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 0.0, 120.0, 25.0)

    mapping = {
        'Male': 1, 'Female': 0,
        'Yes': 1, 'No': 0,
        'No phone service': 2,
        'DSL': 0, 'Fiber optic': 1, 'No': 2,
        'Month-to-month': 0, 'One year': 1, 'Two year': 2
    }

    features = [
        mapping[gender],
        mapping[senior],
        tenure,
        mapping[phone],
        mapping[multi_lines],
        mapping[internet],
        mapping[contract],
        monthly_charges
    ]
    
    return pd.DataFrame([features], columns=X.columns)

input_df = user_input_features()

# Predictions
lr_pred = lr.predict(input_df)[0]
dt_pred = dt.predict(input_df)[0]
svm_pred = svm_model.predict(input_df)[0]

st.subheader("🔍 Predictions:")
st.success(f"🔹 Logistic Regression: {'Churn' if lr_pred == 1 else 'No Churn'}")
st.success(f"🔹 Decision Tree: {'Churn' if dt_pred == 1 else 'No Churn'}")
st.success(f"🔹 SVM: {'Churn' if svm_pred == 1 else 'No Churn'}")

# Accuracy metrics
st.markdown("---")
st.subheader("📈 Model Accuracy (Test Set)")
st.write(f"✅ Logistic Regression: {accuracy_score(y_test, lr.predict(X_test)):.2f}")
st.write(f"✅ Decision Tree: {accuracy_score(y_test, dt.predict(X_test)):.2f}")
st.write(f"✅ SVM: {accuracy_score(y_test, svm_model.predict(X_test)):.2f}")