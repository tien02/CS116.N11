# HovaTen: Dang Anh Tien
# MSSV: 20520800
import time
from turtle import color
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Introduction
st.markdown("# Welcome to 20520800's web 👏👏👏 ")
st.write("Let 20520800 train a superior Linear Regression model for you 😇")

## Dataset
st.markdown("## Dataset 🤗")
st.write("Please give 20520800 the data, 20520800 need data 🥺")
file_uploaded = st.file_uploader("Choose a file")
## Upload file.csv
if file_uploaded:
    df = pd.read_csv(file_uploaded)

    show_dataframe = st.checkbox("Show dataframe")
    if show_dataframe:
        st.dataframe(df)   
    
    columns = df.columns.tolist()
    columns.remove("Profit")

# Select input feature
st.markdown("## Select Input features 🤗")
st.write("Please select which input feature you want 🥺")
try: 
    input_feature = st.multiselect("Input feature", columns)
    ## Process categorical feature
    if input_feature:
        y = df["Profit"].values
        if "State" in input_feature:
            input_feature.remove("State")
            one_hot = OneHotEncoder(handle_unknown='ignore')
            X_categories = df["State"].values.reshape(-1, 1)
            X_categories = one_hot.fit_transform(X_categories).toarray()
            X_cont = df[input_feature].values

            X = np.concatenate([X_cont, X_categories], axis=1)
        else:
            X = df[input_feature].values
except:
    st.write("Files haven't uploaded.")
    pass

tab1, tab2 = st.tabs(["Train-Test split", "K-Folds Cross Validation"])

## Tab1: Train-Test split
with tab1:
    try:
        st.markdown("## Train Test Split 🤗")
        st.write("Please select the ratio of training set you want 🥺")
        ratio = st.slider('Choose ratio', min_value=0, max_value=100, value=80)
        ## Try splitting dataset respecting to the ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio / 100, random_state=42)
        st.markdown("## Training and Testing 🤗")
        st.write("It's training time 🤩, let 20520800 do that for you 😇")
        train_button = st.button("Train", key=1)
        if train_button:
            with st.spinner("Waiting for training..."):
                start_training = time.time()    
                model1 = LinearRegression(n_jobs=4)
                model1.fit(X_train, y_train)
                y_pred1 = model1.predict(X_test)
                end_training = time.time() - start_training
                time.sleep(1)
                st.success(f"Training finished. It takes {end_training:.3f} seconds to complete")
            mse = mean_squared_error(y_test, y_pred1)
            mae = mean_absolute_error(y_test, y_pred1)
            lab = ["MAE", "RMSE"]
            val = [mae, np.sqrt(mse)]
            fig, ax = plt.subplots()
            ax.bar(lab, val, width=0.5, color="green")
            fig.tight_layout()
            st.pyplot(fig)
    except:
        pass

## Tab2: K-Folds validation
with tab2:
    try:
        st.markdown("## K_Folds cross validation 🤗")
        st.write("Please select 'k' for K Folds cross validation 🥺")
        k = st.number_input("Get 'k'", min_value=2, value=10)
        k_fold = KFold(n_splits=int(k), shuffle=True)
        if input_feature:
            if k:
                fold_list = [*range(1, int(k)+1)]
                label_list = ["Fold " + str(f) for f in fold_list ]
                st.markdown("## Training and Testing 🤗")
                st.write("It's training time 🤩, let 20520800 do that for you 😇")
                train_button = st.button("Train", key=2)
                if train_button:
                    mse_list = []
                    mae_list = []
                    with st.spinner("Waiting for training..."):
                        start_training = time.time()    
                        for train_idx, val_idx in k_fold.split(X, y):
                            model2 = LinearRegression(n_jobs=4)
                            model2.fit(X[train_idx], y[train_idx])
                            y_pred = model2.predict(X[val_idx])
                            mse_list.append(np.sqrt(mean_squared_error(y[val_idx], y_pred)))
                            mae_list.append(mean_absolute_error(y[val_idx], y_pred))
                        end_training = time.time() - start_training
                        time.sleep(1)
                        st.success(f"Training finished. It takes {end_training:.3f} seconds to complete")
                    x = np.arange(k)
                    width = 0.35

                    fig, ax = plt.subplots()
                    rects1 = ax.bar(x - width/2, mae_list, width, label='MAE')
                    rects2 = ax.bar(x + width/2, mse_list, width, label='RMSE')

                    # Add some text for labels, title and custom x-axis tick labels, etc.
                    ax.set_title('K-Folds Cross Validation')
                    ax.set_xticks(x, label_list)
                    ax.legend()

                    fig.tight_layout()
                    st.pyplot(fig)
    except:
        pass