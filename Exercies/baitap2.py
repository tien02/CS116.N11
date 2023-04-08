# import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_path = "Regression_dataset\Salary_Data.csv"
df = pd.read_csv(data_path)
X = df["YearsExperience"].values
y = df["Salary"].values

X = np.expand_dims(X, axis = 1)
print(f"X shape: {X.shape}")
print(f"y shape {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

l2 = np.mean(np.square(y_test - y_pred)) ** 0.5
l1 = np.mean(np.absolute(y_test - y_pred))
print(f"L2 loss: {l2:.3f}")
print(f"L1 loss: {l1:.3f}")

plt.scatter(X, y)
plt.plot(X_test, y_pred)
plt.show()

print("Finish!")