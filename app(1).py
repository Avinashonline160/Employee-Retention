import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
st.set_page_config(page_title="Employee Retention Dashboard",layout="wide")
st.title("Employee Retention Prediction")
st.markdown("[My GitHub](https://github.com/Avinashonline160)")
try:
    df=pd.read_csv("HR_comma_sep.csv")
except:
    df=pd.read_excel("HR_comma_sep.xls")
st.dataframe(df.head())
fig,ax=plt.subplots(); pd.crosstab(df["salary"],df["left"]).plot(kind="bar",ax=ax); st.pyplot(fig)
dept="Department" if "Department" in df.columns else "department"
fig,ax=plt.subplots(); pd.crosstab(df[dept],df["left"]).plot(kind="bar",ax=ax); st.pyplot(fig)
X=df[["satisfaction_level","average_montly_hours","promotion_last_5years","salary"]].copy()
y=df["left"]; le=LabelEncoder(); X["salary"]=le.fit_transform(X["salary"])
Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.3,random_state=42)
m=LogisticRegression(max_iter=1000); m.fit(Xtr,Ytr)
st.success(f"Accuracy: {accuracy_score(Yte,m.predict(Xte))*100:.2f}%")
