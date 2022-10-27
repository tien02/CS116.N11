# HovaTen: Dang Anh Tien
# MSSV: 20520800
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    
# Train-Test split
st.markdown("## Train Test Split 🤗")
st.write("Please select the ratio of training set you want 🥺")
ratio = st.slider('Choose ratio', min_value=0, max_value=100, value=80)
## Try splitting dataset respecting to the ratio
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio / 100, random_state=42)
except:
    pass

## K-Folds validation
st.markdown("## K_Folds cross validation 🤗")
st.write("Please select 'k' for K Folds cross validation 🥺")
k = st.number_input("Get 'k'", min_value=2, value=10)
k_fold = KFold(n_splits=int(k), shuffle=True)
fold_list = [*range(1, int(k)+1)]

st.markdown("## Training and Testing 🤗")
st.write("It's training time 🤩, let 20520800 do that for you 😇")
try:
    train_button = st.button("Train")
    if train_button:
        mse_list = []
        mae_list = []
        r2_list = []
        with st.spinner("Waiting for training..."):
            start_training = time.time()    
            model1 = LinearRegression(n_jobs=4)
            model1.fit(X_train, y_train)
            y_pred1 = model1.predict(X_test)

            for train_idx, val_idx in k_fold.split(X_train, y_train):
                model2 = LinearRegression(n_jobs=4)
                model2.fit(X_train[train_idx], y_train[train_idx])
                y_pred = model2.predict(X_train[val_idx])
                mse_list.append(round(mean_squared_error(y_train[val_idx], y_pred), 5))
                mae_list.append(round(mean_absolute_error(y_train[val_idx], y_pred), 5))
                r2_list.append(round(r2_score(y_train[val_idx], y_pred), 5))
            end_training = time.time() - start_training
            time.sleep(1)
            st.success(f"Training finished. It takes {end_training:.3f} seconds to complete")
        st.markdown("### Evaluation on Test set")
        st.metric("MSE on Test set", round(mean_squared_error(y_test, y_pred1), 3))
        st.metric("MAE on Test set",round(mean_absolute_error(y_test, y_pred1), 3))
        st.metric("R2-score on Test set", round(r2_score(y_test, y_pred1), 3))
        eval_dict = {
            "Fold": fold_list,
            "MSE": mse_list,
            "MAE": mae_list,
            "R2-score": r2_list
        }
        eval_df = pd.DataFrame(data=eval_dict)
        st.markdown("### Evaluation on K Folds Cross Validation")
        st.dataframe(eval_df)

        st.metric("Mean MSE K Fold cross validation", round(np.mean(mse_list), 3))
        st.metric("Mean MAE K Fold cross validation", round(np.mean(mae_list), 3))
        st.metric("Mean R2-score K Fold cross validation", round(np.mean(r2_list), 3))
        
except:
    st.error("Get the data completely first.")
    pass