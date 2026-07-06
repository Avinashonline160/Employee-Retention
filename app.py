import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Page Configuration & Title
st.set_page_config(page_title="Employee Retention Dashboard", layout="wide", page_icon="👔")

st.title("👔 Employee Retention Prediction & Analysis Dashboard")
st.markdown("### Developed by [Avinash](https://github.com/Avinashonline160)")
st.write("Predict whether an employee will stay or leave based on HR analytics, and explore workplace insights.")
st.markdown("---")

# 2. Data Loading with Cache to prevent re-running on every input widget change
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("HR_comma_sep.csv")
    except Exception:
        try:
            data = pd.read_excel("HR_comma_sep.xls")
        except Exception:
            st.error("Could not find 'HR_comma_sep.csv' or 'HR_comma_sep.xls' in the current directory. Please make sure the data file is uploaded to GitHub.")
            return None
    return data

df = load_data()

if df is not None:
    # Identify department column variance capitalization
    dept_col = "Department" if "Department" in df.columns else "department"

    # 3. Sidebar for Live Prediction Inputs
    st.sidebar.header("🎯 Live Prediction Input")
    st.sidebar.write("Adjust features below to predict employee retention:")
    
    # Input widgets based on model feature dependencies
    satisfaction = st.sidebar.slider("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    monthly_hours = st.sidebar.number_input("Average Monthly Hours", min_value=90, max_value=320, value=200)
    promotion = st.sidebar.selectbox("Promoted in Last 5 Years?", options=["No", "Yes"])
    salary_level = st.sidebar.selectbox("Salary Category", options=["low", "medium", "high"])

    # Convert readable formats to original binary representation for prediction mapping
    promotion_val = 1 if promotion == "Yes" else 0

    # 4. Main App Layout Split into Tabs
    tab1, tab2 = st.tabs(["📊 Explanatory Data Analysis", "🔮 Machine Learning Prediction Model"])

    with tab1:
        st.subheader("🗂️ Dataset Sample Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 Attrition Visualization Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Retention Status Based on Salary Tier")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            pd.crosstab(df["salary"], df["left"]).plot(kind="bar", stacked=True, ax=ax1, color=["#2ecc71", "#e74c3c"])
            ax1.set_ylabel("Number of Employees")
            ax1.set_xlabel("Salary Tier")
            ax1.legend(["Stayed (0)", "Left (1)"])
            plt.xticks(rotation=0)
            st.pyplot(fig1)

        with col2:
            st.markdown("#### Retention Status Based on Department")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            pd.crosstab(df[dept_col], df["left"]).plot(kind="bar", stacked=True, ax=ax2, color=["#3498db", "#e67e22"])
            ax2.set_ylabel("Number of Employees")
            ax2.set_xlabel("Department")
            ax2.legend(["Stayed (0)", "Left (1)"])
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig2)

    with tab2:
        st.subheader("🧠 Model Training & Prediction Insights")
        
        # Preparing Model Vectors
        X = df[["satisfaction_level", "average_montly_hours", "promotion_last_5years", "salary"]].copy()
        y = df["left"]
        
        # Enforcing identical transformation state structure
        le = LabelEncoder()
        # Custom fit ordering to ensure low=0, medium=1, high=2 map consistently
        le.fit(["low", "medium", "high"]) 
        X["salary"] = le.transform(X["salary"])
        
        Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(Xtr, Ytr)
        
        acc = accuracy_score(Yte, model.predict(Xte)) * 100
        st.info(f"⚡ Current Logistic Regression Model Test Accuracy Score: **{acc:.2f}%**")
        
        st.markdown("---")
        st.subheader("💡 Operational Status Output")
        
        # Format the user inputs from sidebar into an inline prediction array
        encoded_salary = le.transform([salary_level])[0]
        user_features = [[satisfaction, monthly_hours, promotion_val, encoded_salary]]
        
        prediction = model.predict(user_features)[0]
        prediction_prob = model.predict_proba(user_features)[0]

        if prediction == 1:
            st.error(f"🚨 **Prediction: This employee is highly likely to LEAVE.**")
            st.write(f"The model calculations evaluate a risk probability of **{prediction_prob[1]*100:.1f}%** turnover chance.")
        else:
            st.success(f"✅ **Prediction: This employee is stable and likely to STAY.**")
            st.write(f"The model calculates a reassurance security percentage of **{prediction_prob[0]*100:.1f}%** alignment.")
    
