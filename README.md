# ML Algorithm Comparison Using Breast Cancer Dataset

## Project Overview

This project compares three machine learning classification algorithms using the **Breast Cancer Dataset** available in scikit-learn.

The algorithms used are:

1. Decision Tree
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Dataset

The project uses the built-in `Breast Cancer Wisconsin Diagnostic` dataset from `sklearn.datasets`.

The dataset contains:

- **569 samples**
- **30 numerical features**
- **2 target classes**

The classes are:

- Malignant
- Benign

## Algorithms

### 1. Decision Tree

A Decision Tree makes predictions by splitting the data into branches based on feature values.

Feature scaling is not required for a Decision Tree.

### 2. K-Nearest Neighbors (KNN)

KNN classifies a new data point based on the classes of its nearest neighbors.

Here, `n_neighbors=5` is used.

Feature scaling is important for KNN, so `StandardScaler` is applied.

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It assumes that the features follow a Gaussian (normal) distribution.

Feature scaling is also applied in this project.

## Data Splitting

The dataset is divided into:

- **80% training data**
- **20% testing data**

The split uses:

- `random_state=42`
- `stratify=y`

Using `stratify=y` helps maintain the class distribution in the training and testing sets.

## Feature Scaling

`StandardScaler` is used for KNN and Gaussian Naive Bayes.

The scaler is fitted only on the training data and then used to transform both training and testing data.

The Decision Tree uses the original, unscaled data.

## Evaluation Metrics

### Accuracy

Accuracy measures the percentage of total predictions that are correct.

**Formula:**

Accuracy = Correct Predictions / Total Predictions

### Precision

Precision measures how many predicted positive cases are actually positive.

### Recall

Recall measures how many actual positive cases were correctly identified.

### F1-Score

F1-Score is the harmonic mean of Precision and Recall.

It is useful when both Precision and Recall are important.

## Confusion Matrix

A confusion matrix shows the number of:

- True Positives
- True Negatives
- False Positives
- False Negatives

A separate confusion matrix is generated for each algorithm.

## Project Structure

```text
ML_Algorithm_Comparison/
│
├── ml_algorithm_comparison.py
└── README.md
```

The Python file is divided into cells using `# %%` markers.

This makes it possible to run the program cell-by-cell in compatible editors such as:

- VS Code
- Spyder
- Jupyter-compatible environments

## Requirements

Install the required Python libraries using:

```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

### Option 1: VS Code

1. Open `ml_algorithm_comparison.py` in VS Code.
2. Make sure the Python extension is installed.
3. Select a Python interpreter.
4. Run each `# %%` cell using the **Run Cell** option.

### Option 2: Run the Complete File

Use:

```bash
python ml_algorithm_comparison.py
```

## Expected Output

The program displays:

1. Dataset shape and class names
2. Training and testing data shapes
3. Comparison table containing:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
4. The algorithm with the highest F1-Score
5. A performance comparison bar graph
6. Confusion matrix graphs for all three algorithms

## Conclusion

The project provides a simple comparison of three commonly used classification algorithms on the Breast Cancer Dataset.

The model with the highest F1-Score is selected as the best-performing algorithm based on the evaluation performed by the program.

**Note:** The exact metric values can depend on the dataset split and model settings. The Python program calculates the values automatically when it is executed.
