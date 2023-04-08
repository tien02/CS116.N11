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
    checks = st.columns(2)

    with checks[0]:
        all = st.checkbox("Select all")
    with checks[1]:
        use_pca = st.checkbox("Use PCA")
    
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

## PCA
try:
    if use_pca:
        st.markdown("## Select Number of PCA components 🤗")
        pca_components = st.number_input("Select PCA components", min_value=1,
        max_value=len(input_feature), value=len(input_feature), step=1)
        st.write("Number of selected PCA components equal ")
except:
    pass

st.markdown("## Training strategy")
tab1, tab2 = st.tabs(["Train-Test split", "K-Folds Cross Validation"])

## Tab1: Train-Test split
with tab1:
    try:
        st.markdown("## Train Test Split 🤗")
        st.write("Please select the ratio of training set you want 🥺")
        ratio = st.slider('Choose ratio', min_value=0, max_value=100, value=80)
        ## Try splitting dataset respecting to the ratio
        scaler1 = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio / 100, random_state=42)
        scaler1.fit(X_train)
        X_train_norm = scaler1.transform(X_train)
        X_test_norm = scaler1.transform(X_test)


        st.markdown("## Training and Testing 🤗")
        st.write("It's training time 🤩, let 20520800 do that for you 😇")
        train_button = st.button("Train", key=1)
        if train_button:
            with st.spinner("Waiting for training..."):
                start_training = time.time()    
                if use_pca:
                    pca = PCA(n_components=pca_components)
                    X_train_comps = pca.fit_transform(X_train_norm)
                    X_test_comps = pca.transform(X_test_norm)
                    model1 = LogisticRegression()
                    model1.fit(X_train_comps, y_train)
                    y_pred1 = model1.predict(X_test_comps)
                    y_pred1_prob = model1.predict_proba(X_test_comps)

                else:
                    model1 = LogisticRegression()
                    model1.fit(X_train_norm, y_train)
                    y_pred1 = model1.predict(X_test_norm)
                    y_pred1_prob = model1.predict_proba(X_test_norm)

                end_training = time.time() - start_training
                time.sleep(1)
                st.success(f"Training finished. It takes {end_training:.3f} second to complete")

            precision = round(precision_score(y_test, y_pred1, average="weighted"), 4)
            recall = round(recall_score(y_test, y_pred1, average="weighted"), 4)
            f1 = round(f1_score(y_test,y_pred1, average="weighted"), 4)
            log = round(log_loss(y_test, y_pred1_prob), 4)

            lab = ["Precision", "Recall", "F1-score", "Log Loss"]
            val = [precision, recall, f1, log]
            fig, ax = plt.subplots()
            bar_plot = ax.bar(lab, val, width=0.4, color="green")
            ax.bar_label(bar_plot, label=val, label_type="center")
            st.pyplot(fig)
    except:
        pass

## Tab2: K-Folds validation
with tab2:
    try:
        st.markdown("## K-Folds Cross Validation 🤗")
        st.write("Please select 'k' for K Folds cross validation 🥺")
        k = st.number_input("Get 'k'", min_value=2, value=5)
        k_fold = KFold(n_splits=int(k), shuffle=True)
        fold_list = [*range(1, int(k)+1)]
        if input_feature:
            if k:
                label_list = ["Fold " + str(f) for f in fold_list ]
                st.markdown("## Training and Testing 🤗")
                st.write("It's training time 🤩, let 20520800 do that for you 😇")
                train_button = st.button("Train", key=2)
                if train_button:
                    scaler2 = StandardScaler()
                    scaler2.fit(X)
                    X_norm = scaler2.transform(X)
                    precision_list = []
                    recall_list = []
                    f1_list = []
                    log_list = []
                    with st.spinner("Waiting for training..."):
                        start_training = time.time()
                        if use_pca:
                            for train_idx, val_idx in k_fold.split(X_norm, y):
                                pca = PCA(n_components=pca_components)
                                X_train_comps = pca.fit_transform(X_norm[train_idx])
                                X_test_comps = pca.transform(X_norm[val_idx])

                                model2 = LogisticRegression(n_jobs=4)
                                model2.fit(X_train_comps, y[train_idx])

                                y_pred = model2.predict(X_test_comps)
                                y_pred_prob = model2.predict_proba(X_test_comps)
                                
                                precision_list.append(round(precision_score(y[val_idx], y_pred, average="weighted"), 4))
                                recall_list.append(round(recall_score(y[val_idx], y_pred, average="weighted"), 4))
                                f1_list.append(round(f1_score(y[val_idx], y_pred, average="weighted"), 4))
                                log_list.append(round(log_loss(y[val_idx], y_pred_prob), 4))
                        else:
                            for train_idx, val_idx in k_fold.split(X_norm, y):
                                model2 = LogisticRegression(n_jobs=4)
                                model2.fit(X_norm[train_idx], y[train_idx])

                                y_pred = model2.predict(X_norm[val_idx])
                                y_pred_prob = model2.predict_proba(X_norm[val_idx])

                                precision_list.append(round(precision_score(y[val_idx], y_pred, average="weighted"), 4))
                                recall_list.append(round(recall_score(y[val_idx], y_pred, average="weighted"), 4))
                                f1_list.append(round(f1_score(y[val_idx], y_pred, average="weighted"), 4))
                                log_list.append(round(log_loss(y[val_idx], y_pred_prob), 4))
                        end_training = time.time() - start_training
                        time.sleep(1)
                        st.success(f"Training finished. It takes {end_training:.3f} seconds to complete")
                    x = np.linspace(0, 2 * k, k)
                    width = 0.3

                    fig, ax = plt.subplots()
                    rects1 = ax.bar(x-0.5, precision_list, width, label='Precision', align="edge")
                    rects2 = ax.bar(x-0.1, recall_list, width, label='Recall')
                    rects3 = ax.bar(x+0.2, f1_list, width, label='F1-score')
                    rects4 = ax.bar(x+0.5, log_list, width, label='Log Loss')

                    # Add some text for labels, title and custom x-axis tick labels, etc.
                    ax.set_title('K-Folds Cross Validation')
                    ax.set_xticks(x, label_list)
                    ax.legend()

                    fig.tight_layout()
                    st.pyplot(fig)

                    mean_precision =  round(np.mean(precision_list),4)
                    mean_recall = round(np.mean(recall_list),4)
                    mean_f1 = round(np.mean(f1_list),4)
                    mean_log_loss = round(np.mean(log_list), 4)

                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    with col_metric1:
                        st.metric("Avg Precision", mean_precision)
                    with col_metric2:
                        st.metric("Avg Recall", mean_recall)
                    with col_metric3:
                        st.metric("Avg F1-score", mean_f1)
                    with col_metric4:
                        st.metric("Avg Log Loss", mean_log_loss)
    except:
        pass