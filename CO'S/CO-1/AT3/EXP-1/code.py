from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual and Predicted labels
actual = ["Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Fail", "Fail"]
predicted = ["Pass", "Fail", "Pass", "Fail", "Fail", "Pass", "Pass", "Fail"]

# Confusion Matrix
cm = confusion_matrix(actual, predicted)

# Display Heatmap
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
