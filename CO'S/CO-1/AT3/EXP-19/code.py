import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

X, y = make_circles(
    n_samples=300,
    factor=0.5,
    noise=0.08,
    random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    learning_rate_init=0.03,
    activation='identity',
    max_iter=2000,
    random_state=42
)

model.fit(X, y)

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

print("Accuracy:", accuracy)

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Experiment 19 - Confusion Matrix")
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y_pred,
    cmap='viridis',
    edgecolor='black'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Circular Data Classification")
plt.show()
