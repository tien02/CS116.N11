'''
Ho va Ten: Dang Anh Tien
Ma so sinh vien: 20520800
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def log_normalize(feature):
    return np.log(feature)

def accuracy(y_true, y_pred):
    return np.mean((y_true == y_pred).astype(int))

def plot_data(X, y, title):
    idx = 1
    plt.figure(figsize=(12, 6))
    plt.suptitle(title)
    for i in range(4):
        for j in range(4):
            if i >= j:
                continue
            plt.subplot(2,3, idx)
            plt.scatter(X[:, i], X[:, j], c = y)
            plt.title(f"Feature {i} and {j}")
            idx += 1
    plt.show()

def plot_predict(X, y, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.suptitle(title)
    plt.subplot(1,2,1)
    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.title("Ground Truth")

    plt.subplot(1,2,2)
    plt.scatter(X[:, 0], X[:, 1], c = y_pred)
    plt.title("Predict")
    plt.show()

if __name__ == '__main__':
    print("\n*** Bài Tập 1 - xử lý dữ liệu ***\n")
    
    # Ex1
    data = load_iris()
    X = data.data
    y = data.target
    class_names = data.target_names

    print("\tTải dữ liệu")
    print("*" * 69)
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Classes: {class_names}")
    print("-" * 69)

    # Ex2
    print("\n\tChia dữ liệu huấn luyện và kiểm tra")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=69)
    print("*" * 69)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("-" * 69)

    # Ex3
    print("\n\tChuẩn hóa dữ liệu")
    print("*" * 69)

    print(f"(Before Normalize) Max: {X_train.max()} | Min: {X_train.min()}")
    plot_data(X_train, y_train, "Before Normalize")

    X_train_norm = log_normalize(X_train)

    print(f"(After Normalize) Max: {X_train_norm.max():.2f} | Min: {X_train_norm.min():.2f}")
    plot_data(X_train_norm, y_train, "After Normalize")

    print("-" * 69)

    # Ex4
    print("\n\tTrực quan hóa dữ liệu")
    print("*" * 69)

    pred_train = np.random.randint(low=0, high=3, size=(X_train.shape[0],))
    plot_predict(X_train, y_train, pred_train, "Train Set")
    print(f"Train accuracy: {accuracy(y_train, pred_train):.2f}")

    pred_test = np.random.randint(low=0, high=3, size=(X_test.shape[0],))
    plot_predict(X_test, y_test, pred_test, "Test Set")
    print(f"Test accuracy: {accuracy(y_test, pred_test):.2f}")
    print("-" * 69)