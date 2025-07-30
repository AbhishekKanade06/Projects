import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  
from Preprocess import Preprocess
from ModelsTrainer import ModelsTrainer 
import time

st.set_page_config(page_title="AutoML System", layout="wide")
st.title("🧠 AutoML Pipeline")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df=pd.read_csv(uploaded_file)
    st.write("### 🧾 Dataset Preview")
    st.dataframe(df.head())
    # Select target
    target = st.selectbox("🎯 Select the target column:", df.columns)
    if target:
        pre= Preprocess(df, target)
        button = st.button("Preprocess Data")
        if button:
            with st.spinner("Processing..."):
                
                X_scaled, y, le, scaler,type = pre.preprocess()
                trainer = ModelsTrainer(X_scaled, y, le, scaler, type)
                res,tuned_model = trainer.train()
                time.sleep(2)
                if res:
                    st.write(res)
            st.success("Data Preprocessed Successfully!")

            