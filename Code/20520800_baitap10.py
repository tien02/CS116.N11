# HovaTen: Dang Anh Tien
# MSSV: 20520800
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics

#evaluate
def evaluate(y_true, y_pred):
    pre = metrics.precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
    re = metrics.recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average="weighted")

    return {
        "pre": pre,
        "re": re,
        "f1":f1
    }

# utils
def convert_class(val):
    if val == 2:
        return 0
    return 1

def plot_multi_metrics(eva_list):
    precision_list = [metric["pre"] for metric in eva_list]
    recall_list = [metric["re"] for metric in eva_list]
    f1_list = [metric["f1"] for metric in eva_list]
    label_list = ["Decision Tree", "Logistic", "SVM", "XGBoost"]

    x = np.linspace(0, 2 * 4, 4)
    width = 0.3

    fig, ax = plt.subplots()
    rects1 = ax.bar(x-0.5, precision_list, width, label='Precision', align="edge")
    rects2 = ax.bar(x-0.1, recall_list, width, label='Recall')
    rects3 = ax.bar(x+0.2, f1_list, width, label='F1-score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Compare Different Models')
    ax.set_xticks(x, label_list)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)

def plot_multi_metrics_kfold(metric_list, model_type):
    metric_arr = np.array(metric_list)

    precision_list = list(metric_arr[:, 0])
    recall_list = list(metric_arr[:, 1])
    f1_list = list(metric_arr[:, 2])
    k = len(precision_list)

    x = np.linspace(0, 2 * k, k)
    width = 0.3

    fig, ax = plt.subplots()
    rects1 = ax.bar(x-0.5, precision_list, width, label='Precision', align="edge")
    rects2 = ax.bar(x, recall_list, width, label='Recall')
    rects3 = ax.bar(x+0.5, f1_list, width, label='F1-score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(model_type + ' K-Folds Cross Validation')
    ax.set_xticks(x, label_list)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)

    mean_precision =  round(np.mean(precision_list),4)
    mean_recall = round(np.mean(recall_list),4)
    mean_f1 = round(np.mean(f1_list),4)

    col_metric1, col_metric2, col_metric3= st.columns(3)
    with col_metric1:
        st.metric("Avg Precision", mean_precision)
    with col_metric2:
        st.metric("Avg Recall", mean_recall)
    with col_metric3:
        st.metric("Avg F1-score", mean_f1)


st.set_page_config(page_title="20520800's website", 
                    page_icon="randomd")

# Introduction
st.markdown("# Welcome to 20520800's web 👏👏👏 ")
st.write("Let 20520800 train a superior Logistic Regression model for you 😇")

## Dataset
st.markdown("## Dataset 🤗")
st.write("Please give 20520800 the data, 20520800 need data 🥺")
file_uploaded = st.file_uploader("Choose a file")

## Upload file.csv
if file_uploaded:
    df = pd.read_csv(file_uploaded, index_col=0)
    df["Class"] = df["Class"].apply(convert_class)

    with st.expander("Show dataframe"):  
        st.dataframe(df)
    
    columns = df.columns.tolist()
    columns.remove("Class")

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
        y = df["Class"].values
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
except:
    pass

logistic = LogisticRegression()
tree = DecisionTreeClassifier()
svm = SVC()
xgboost = XGBClassifier(n_estimators=100, 
                    max_depth=2,
                    learning_rate=1, 
                    objective='binary:logistic')

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

                    tree.fit(X_train_comps, y_train)
                    svm.fit(X_train_comps, y_train)
                    xgboost.fit(X_train_comps, y_train)
                    logistic.fit(X_train_comps, y_train)


                    tree_pred = tree.predict(X_test_comps)
                    svm_pred = svm.predict(X_test_comps)
                    xgboost_pred = xgboost.predict(X_test_comps)
                    logistic_pred = logistic.predict(X_test_comps)
                else:
                    tree.fit(X_train_norm, y_train)
                    svm.fit(X_train_norm, y_train)
                    xgboost.fit(X_train_norm, y_train)
                    logistic.fit(X_train_norm, y_train)

                    tree_pred = tree.predict(X_test_norm)
                    svm_pred = svm.predict(X_test_norm)
                    xgboost_pred = xgboost.predict(X_test_norm)
                    logistic_pred = logistic.predict(X_test_norm)

                end_training = time.time() - start_training
                time.sleep(1)
                st.success(f"Training finished. It takes {end_training:.3f} second to complete")

            tree_eva = evaluate(y_test, tree_pred)
            svm_eva = evaluate(y_test, svm_pred)
            xgboost_eva = evaluate(y_test, xgboost_pred)
            logistic_eva = evaluate(y_test, logistic_pred)

            eva_list = [tree_eva, logistic_eva, svm_eva, xgboost_eva]

            plot_multi_metrics(eva_list)
    except:
        pass

## Tab2: K-Folds validation
with tab2:
    try:
        st.markdown("## K-Folds Cross Validation 🤗")
        st.write("Please select 'k' for K Folds cross validation 🥺")
        k = st.number_input("Get 'k'", min_value=2, value=5)
        k_fold = KFold(n_splits=int(k), shuffle=True, random_state=42)
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

                    metrics_dict = {
                        "tree": [],
                        "svm": [],
                        "xgboost": [],
                        "logistic": []
                    }

                    with st.spinner("Waiting for training..."):
                        start_training = time.time()
                        if use_pca:
                            for train_idx, val_idx in k_fold.split(X_norm, y):
                                pca = PCA(n_components=pca_components)
                                X_train_comps = pca.fit_transform(X_norm[train_idx])
                                X_test_comps = pca.transform(X_norm[val_idx])

                                tree.fit(X_train_comps, y[train_idx])
                                svm.fit(X_train_comps, y[train_idx])
                                xgboost.fit(X_train_comps, y[train_idx])
                                logistic.fit(X_train_comps, y[train_idx])


                                tree_pred = tree.predict(X_test_comps)
                                svm_pred = svm.predict(X_test_comps)
                                xgboost_pred = xgboost.predict(X_test_comps)
                                logistic_pred = logistic.predict(X_test_comps)

                                tree_eva = evaluate(y[val_idx], tree_pred)
                                svm_eva = evaluate(y[val_idx], svm_pred)
                                xgboost_eva = evaluate(y[val_idx], xgboost_pred)
                                logistic_eva = evaluate(y[val_idx], logistic_pred)
 
                                metrics_dict["tree"].append(list(tree_eva.values()))
                                metrics_dict["svm"].append(list(svm_eva.values()))
                                metrics_dict["xgboost"].append(list(xgboost_eva.values()))
                                metrics_dict["logistic"].append(list(logistic_eva.values()))
    
                        else:
                            for train_idx, val_idx in k_fold.split(X_norm, y):
                                tree.fit(X_norm[train_idx], y[train_idx])
                                svm.fit(X_norm[train_idx], y[train_idx])
                                xgboost.fit(X_norm[train_idx], y[train_idx])
                                logistic.fit(X_norm[train_idx], y[train_idx])


                                tree_pred = tree.predict(X_norm[val_idx])
                                svm_pred = svm.predict(X_norm[val_idx])
                                xgboost_pred = xgboost.predict(X_norm[val_idx])
                                logistic_pred = logistic.predict(X_norm[val_idx])

                                tree_eva = evaluate(y[val_idx], tree_pred)
                                svm_eva = evaluate(y[val_idx], svm_pred)
                                xgboost_eva = evaluate(y[val_idx], xgboost_pred)
                                logistic_eva = evaluate(y[val_idx], logistic_pred)
 
                                metrics_dict["tree"].append(list(tree_eva.values()))
                                metrics_dict["svm"].append(list(svm_eva.values()))
                                metrics_dict["xgboost"].append(list(xgboost_eva.values()))
                                metrics_dict["logistic"].append(list(logistic_eva.values()))
                        end_training = time.time() - start_training
                        time.sleep(1)
                        st.success(f"Training finished. It takes {end_training:.3f} seconds to complete")

                    # Plot
                    plot_multi_metrics_kfold(metric_list=metrics_dict["tree"], model_type="TREE")

                    # Logistic Regression
                    plot_multi_metrics_kfold(metric_list=metrics_dict["logistic"], model_type="LOGISTIC")

                    # SVM
                    plot_multi_metrics_kfold(metric_list=metrics_dict["svm"], model_type="SVM")

                    # XGBoost
                    plot_multi_metrics_kfold(metric_list=metrics_dict["xgboost"], model_type="XGBOOST")

    except:
        pass