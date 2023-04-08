# HovaTen: Dang Anh Tien
# MSSV: 20520800
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss

# Introduction
st.markdown("# Welcome to 20520800's web 👏👏👏 ")
st.write("Let 20520800 train a superior Logistic Regression model for you 😇")

## Dataset
st.markdown("## Dataset 🤗")
st.write("Please give 20520800 the data, 20520800 need data 🥺")
file_uploaded = st.file_uploader("Choose a file")

## Upload file.csv
if file_uploaded:
    df = pd.read_csv(file_uploaded)

    with st.expander("Show dataframe"):  
        st.dataframe(df)
    
    columns = df.columns.tolist()
    columns.remove("Wine")

## Select input feature
st.markdown("## Select Input features 🤗")
st.write("Please select which input feature you want 🥺")
try: 

    container = st.container()
    all = st.checkbox("Select all")
    if all:
        input_feature = container.multiselect("Select input feature:", columns, columns)
    else:
        input_feature = container.multiselect("Select input feature:", columns)
    if input_feature:
        y = df["Wine"].values
        X = df[input_feature].values

except:
    st.write("Files haven't uploaded.")
    pass    

try:
    st.markdown("## Select Number of PCA components 🤗")
    pca_components = st.number_input("Select PCA components", min_value=1,
    max_value=len(input_feature), value=len(input_feature), step=1)

    st.markdown("## K-Folds Cross Validation 🤗")
    st.write("Please select 'k' for K Folds cross validation 🥺")
    k = st.number_input("Get 'k'", min_value=2, value=5)
    k_fold = KFold(n_splits=int(k), shuffle=True, random_state=42)
    fold_list = [*range(1, int(k)+1)]
    if input_feature:
        if k:
            st.markdown("## Training and Testing to find PCA components for best F1-score🤗")
            st.write("It's training time 🤩, let 20520800 do that for you 😇")
            train_button = st.button("Train", key=2)
            if train_button:
                scaler2 = StandardScaler()
                scaler2.fit(X)
                X_norm = scaler2.transform(X)
                with st.spinner("Waiting for training..."):
                    f1_pca_component = {}
                    for pca_component in range(1, pca_components+1):
                        f1_pca_list = []
                        for train_idx, val_idx in k_fold.split(X_norm, y):
                            pca = PCA(n_components=pca_component, random_state=42)
                            X_train_comps = pca.fit_transform(X_norm[train_idx])
                            X_test_comps = pca.transform(X_norm[val_idx])

                            model2 = LogisticRegression(n_jobs=4)
                            model2.fit(X_train_comps, y[train_idx])

                            y_pred = model2.predict(X_test_comps)
                            y_pred_prob = model2.predict_proba(X_test_comps)

                            f1_pca_list.append(f1_score(y[val_idx], y_pred, average="weighted"))
                        f1_pca_component[pca_component] = np.mean(f1_pca_list)
                    fig, ax = plt.subplots()
                    
                    lab = f1_pca_component.keys()
                    val = f1_pca_component.values()

                    bar_plot = ax.bar(lab, val, width=0.6, color="green")
                    ax.set_xlabel("PCA n_components")
                    ax.set_ylabel("F1-score")
                    ax.set_title("PCA-components for F1-score")
                    st.pyplot(fig)

                    max_f1_pca = max(f1_pca_component, key= lambda x: f1_pca_component[x])
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("Number of PCA Components", max_f1_pca)
                    with col_metric2:
                        st.metric("Max F1-score", round(f1_pca_component.get(max_f1_pca),4))
except:
    pass