#TASK1 Decision Tree Implementation
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Create a DataFrame with feature names
df = pd.DataFrame(X, columns=wine.feature_names)

# Add the target column
df['target'] = y

# Save to CSV
df.to_csv("wine_dataset.csv", index=False)

print("Wine dataset saved as 'wine_dataset.csv'")

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# Convert to a DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

#PREPROCESSING
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Feature scaling (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#TRAIN THE MODEL
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

#PREDICTIONS & METRICS
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on Test Set: {accuracy:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

# Confusion matrix visualization using matplotlib
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)

# Labeling the cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


# 1. Plotting distribution of each feature
plt.figure(figsize=(16, 20))
for i, col in enumerate(feature_names):
    plt.subplot(5, 3, i + 1)
    for label in np.unique(y):
        plt.hist(df[df['target'] == label][col], alpha=0.5, label=target_names[label], bins=15)
    plt.title(f"Feature: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend()
plt.tight_layout()
plt.show()

# 2. Feature Importance Plot
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.title("Feature Importances")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.tight_layout()
plt.show()
