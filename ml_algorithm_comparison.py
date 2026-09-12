# ============================================================
# ML ALGORITHM COMPARISON
# Dataset: Breast Cancer Dataset
# Algorithms: Decision Tree, KNN, Gaussian Naive Bayes
# ============================================================

# %% [Cell 1] Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# %% [Cell 2] Load Dataset
# Load the Breast Cancer dataset
data = load_breast_cancer()

# Store features and target values
X = data.data
y = data.target

# Display dataset information
print("Dataset Shape:", X.shape)
print("Classes:", data.target_names)

# %% [Cell 3] Split Dataset
# Split dataset into training and testing data
# 80% is used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# %% [Cell 4] Scaling
# StandardScaler converts features to a common scale
scaler = StandardScaler()

# Fit scaler on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# %% [Cell 5] Create Models
# Create three machine learning classification models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes": GaussianNB()
}

# %% [Cell 6] Train and Evaluate
results = []

# Train and test each algorithm
for name, model in models.items():

    # Decision Tree does not require feature scaling
    if name == "Decision Tree":
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    # KNN and Naive Bayes use scaled data
    else:
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)

    # Calculate performance metrics
    results.append([
        name,
        accuracy_score(y_test, prediction),
        precision_score(y_test, prediction),
        recall_score(y_test, prediction),
        f1_score(y_test, prediction)
    ])

# %% [Cell 7] Display Results
# Convert results into a DataFrame
results = pd.DataFrame(
    results,
    columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score"]
)

# Display algorithm comparison
print("\n===== ALGORITHM COMPARISON =====")
print(results.round(4).to_string(index=False))

# %% [Cell 8] Best Algorithm
# Find the algorithm with the highest F1-Score
best = results.loc[results["F1-Score"].idxmax()]

print("\n===== BEST ALGORITHM =====")
print("Algorithm:", best["Algorithm"])
print("F1-Score:", round(best["F1-Score"], 4))

# %% [Cell 9] Performance Graph
# Create a bar graph comparing all algorithms
results.set_index("Algorithm").plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("ML Algorithm Performance")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [Cell 10] Confusion Matrix Graphs
# Generate confusion matrix for each algorithm
for name, model in models.items():

    # Train and predict using the appropriate data
    if name == "Decision Tree":
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, prediction)

    # Display confusion matrix as a graph
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=data.target_names
    )

    disp.plot()

    # Add title to the graph
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()
    plt.show()
