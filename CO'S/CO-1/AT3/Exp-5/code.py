import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data (Study Hours vs Marks)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 65, 75])

# Create Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Predict Marks
y_pred = model.predict(X)

# Predict for a new value
new_hours = np.array([[7]])
predicted_marks = model.predict(new_hours)

# Print Prediction
print("Predicted Marks for 7 Hours:", predicted_marks[0])

# Plot Graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.legend()
plt.show()
