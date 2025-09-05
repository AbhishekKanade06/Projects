import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io, time, pickle,zipfile,os

from Preprocess import Preprocess
from ModelsTrainer import ModelsTrainer
from ScriptGenerator import ScriptGenerator
from sklearn.metrics import classification_report, r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------
# PAGE & STYLE SETTINGS
# ----------------------------------------------------
st.set_page_config(page_title="AutoML Pro", layout="wide", page_icon="🤖")
st.markdown("""
<style>
h1, .main-title { color: #E91E63; font-size: 2.7rem; font-weight: 700; margin-bottom: 0.2em; }
.section-card { background: #22223B; border-radius: 18px; padding: 2.5rem; margin-bottom: 1.5rem; box-shadow: 0 0 16px rgba(0,0,0,0.13);}
.metric-badge { color: #10D876; font-size: 1.25rem; font-weight: 600;}
.success-section {background: #36D39922; border-radius: 1rem; padding: 0.7rem 1.5rem;}
.sidebar-title { color: #E91E63; font-size:1.25rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 AutoML Pro — Seamless Data to Model</div>', unsafe_allow_html=True)
st.write("Upload, preprocess, visualize data, and let AI select the best model — all with an elegant interface.")

# ----------------------------------------------------
# SIDEBAR SETTINGS
# ----------------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">Step 1: Upload Data</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload your dataset (.csv)", type=["csv"])
example = st.sidebar.checkbox("Use Example Dataset")
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-title">Step 2: Select Target</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# DATA UPLOAD & TARGET SELECTION
# ----------------------------------------------------
if example:
    # Example dataset: Iris
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)
    target = "target"
    st.session_state.df = df
    st.session_state.target = target
elif uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.target = None
else:
    st.session_statedf = None

if st.session_state.df is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📋 Dataset Preview")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    st.markdown(f"**Shape:** {st.session_state.df.shape[0]} Rows × {st.session_state.df.shape[1]} Columns")
    st.markdown('</div>', unsafe_allow_html=True)
    target_col = st.selectbox("🎯 Select Target Column", st.session_state.df.columns, key="target_col", index=(st.session_state.df.columns.get_loc(st.session_state.target) if st.session_state.target else 0))
else:
    st.info("Upload a CSV file or load the example dataset to start.")
    target_col = None

# -----------------------------------------------------
# SECTION: PREPROCESSING
# -----------------------------------------------------
if df is not None and target_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Data Preprocessing")
    if st.button("Start Preprocessing", type="primary"):
        with st.spinner("Preprocessing..."):
            pre = Preprocess(df, target_col)
            X_scaled, y, le, scaler, model_type = pre.preprocess()
            st.session_state.X_scaled = X_scaled
            st.session_state.y = y
            st.session_state.le = le
            st.session_state.scaler = scaler
            st.session_state.model_type = model_type
            st.session_state.preprocessed_df = X_scaled.copy()
            st.session_state.preprocessed_df[target_col] = y
            time.sleep(0.5)
        st.success("Preprocessing completed!")
    if "preprocessed_df" in st.session_state:
        st.write("#### Preprocessed Data Sample")
        st.dataframe(st.session_state.preprocessed_df.head())
        buffer = io.BytesIO()
        st.session_state.preprocessed_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("⬇️ Download Preprocessed CSV", buffer, file_name="preprocessed_data.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# SECTION: VISUALIZATION
# -----------------------------------------------------
if "model_type" in st.session_state:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Target Visualization")
    if st.session_state.model_type == 'categorical':
        fig, ax = plt.subplots()
        sns.countplot(x=st.session_state.y, ax=ax, palette="viridis")
        ax.set_title('Target Class Distribution')
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.histplot(st.session_state.y, kde=True, ax=ax, color="#F63366")
        ax.set_title('Target Value Distribution')
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# SECTION: MODEL TRAINING & EVALUATION
# -----------------------------------------------------
if st.session_state.get("X_scaled") is not None and st.session_state.get("y") is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🚀 Model Training and Selection")
    if st.button("Train & Tune Models", type="primary"):
        with st.spinner("Training all models and tuning hyperparameters..."):
            trainer = ModelsTrainer(
                st.session_state.X_scaled,
                st.session_state.y,
                st.session_state.le,
                st.session_state.scaler,
                st.session_state.model_type,
            )
            results, best_model = trainer.train()
            st.session_state.best_model = best_model
            st.session_state.results = results
            st.session_state.best_name = max(results, key=results.get)
            time.sleep(1)
        st.success("Model training and selection complete!")

    if "results" in st.session_state:
        st.write("##### Models & Scores")
        st.table(pd.DataFrame([st.session_state.results]).T.rename(columns={0:"Score"}).sort_values("Score", ascending=False))
        st.markdown(f'**🏅 Best Model:** <span class="metric-badge">{st.session_state.best_name}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------
    # MODEL METRICS & VISUALS
    # -----------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if st.session_state.model_type == 'categorical' and "best_model" in st.session_state:
        st.write("#### 📊 Classification Report")
        y_pred = st.session_state.best_model.predict(st.session_state.X_scaled)
        st.text(classification_report(st.session_state.y, y_pred))
        st.write("#### Confusion Matrix")
        fig, ax = plt.subplots()
        cm = pd.crosstab(pd.Series(st.session_state.y, name='Actual'), pd.Series(y_pred, name='Predicted'))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
    elif "best_model" in st.session_state:
        st.write("#### 📈 Regression Metrics")
        y_pred = st.session_state.best_model.predict(st.session_state.X_scaled)
        st.write(f"**R² Score:** <span class='metric-badge'>{r2_score(st.session_state.y, y_pred):.4f}</span>", unsafe_allow_html=True)
        st.write(f"**MAE:** <span class='metric-badge'>{mean_absolute_error(st.session_state.y, y_pred):.4f}</span>", unsafe_allow_html=True)
        st.write(f"**MSE:** <span class='metric-badge'>{mean_squared_error(st.session_state.y, y_pred):.4f}</span>", unsafe_allow_html=True)
        st.write("#### Actual vs. Predicted")
        fig, ax = plt.subplots()
        sns.scatterplot(x=st.session_state.y, y=y_pred, ax=ax, color="#10D876")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------
    # MODEL DOWNLOAD
    # -----------------------------------------------
    
    st.markdown('<div class="success-section">', unsafe_allow_html=True)
    if "best_model" in st.session_state:
        sg=ScriptGenerator(df,target_col)
        zip_buffer=io.BytesIO()
        with zipfile.ZipFile(zip_buffer,'a',zipfile.ZIP_DEFLATED) as zip_file:
            sg.create_script()
            zip_file.writestr('script.py',sg.script)
            zip_file.writestr('tuned_model.pkl',pickle.dumps(st.session_state.best_model))
            zip_file.writestr('label.pkl',pickle.dumps(st.session_state.le))
            zip_file.writestr('scaler.pkl',pickle.dumps(st.session_state.scaler))
        st.download_button(
            "⬇️ Download Best Model",
            data=zip_buffer.getvalue(),
            file_name="best_model.zip",
            mime="application/zip"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# FOOTER INFO
# -----------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 16px; color: #555; padding-top: 30px;">
        <p style="font-family: 'Arial', sans-serif; font-weight: bold; font-size: 20px; color: #6A5ACD; animation: fadeIn 2s ease-in-out;">
            Crafted with 💎 and Code
        </p>
        <p style="font-family: 'Courier New', monospace; font-size: 14px; color: #888; animation: fadeIn 2s ease-in-out 0.5s;">
            Empowering AutoML workflows with Streamlit 🚀
        </p>
    </div>
    
    <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, 
    unsafe_allow_html=True
)


