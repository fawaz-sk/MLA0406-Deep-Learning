import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7])

# Polynomial Features (High Degree)
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

# Train Model
model = LinearRegression()
model.fit(X_poly, y)

# Prediction
y_pred = model.predict(X_poly)

# Plot
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, y_pred, color='red', label='Overfitting Curve')
plt.title("Overfitting Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
