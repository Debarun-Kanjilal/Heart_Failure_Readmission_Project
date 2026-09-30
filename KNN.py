# ============================================================
# HEART FAILURE 30-DAY READMISSION PREDICTION
# K-NEAREST NEIGHBOURS (KNN)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# 1. DATA LOADING
# ============================================================

print("=" * 70)
print("1. DATA LOADING")
print("=" * 70)

DATA_PATH = "data/dataset_12000_records.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Shape of dataset:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# 2. DEFINING FEATURES AND TARGET VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("2. DEFINING FEATURES AND TARGET VARIABLES")
print("=" * 70)

TARGET = "Readmitted_30_Days"

X = df.drop(columns=[TARGET, "Patient_ID"])

y = df[TARGET]

print("Target variable:", TARGET)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 3. DATA PROCESSING
# ============================================================

print("\n" + "=" * 70)
print("3. DATA PROCESSING")
print("=" * 70)

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# Numerical processing
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical processing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Combine processing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 4. MODEL DEFINITION AND TRAINING
# ============================================================

print("\n" + "=" * 70)
print("4. MODEL DEFINITION AND TRAINING")
print("=" * 70)

# K value
K = 7

knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=K
            )
        )
    ]
)

print(f"K value: {K}")

print("\nModel:")
print(knn_model)

# Train model
knn_model.fit(X_train, y_train)

print("\nKNN model trained successfully.")


# ============================================================
# 5. MAKING PREDICTIONS AND EVALUATING THE MODEL
# ============================================================

print("\n" + "=" * 70)
print("5. MAKING PREDICTIONS AND EVALUATING THE MODEL")
print("=" * 70)

# Predictions
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)

# Probability predictions
y_test_prob = knn_model.predict_proba(X_test)[:, 1]


# Metrics

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)

train_test_gap = train_accuracy - test_accuracy


# Display results

print("\nMODEL PERFORMANCE")
print("-" * 50)

print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1-score          : {f1:.4f}")
print(f"ROC-AUC           : {roc_auc:.4f}")
print(f"Train-Test Gap    : {train_test_gap:.4f}")


# Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_test_pred
)

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX")
print("-" * 50)

print("True Negative  :", tn)
print("False Positive :", fp)
print("False Negative :", fn)
print("True Positive  :", tp)


# Classification Report

print("\nCLASSIFICATION REPORT")
print("-" * 50)

print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Readmitted", "Readmitted"],
    yticklabels=["Not Readmitted", "Readmitted"]
)

plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/figures/knn_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_test_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"KNN (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("KNN - ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/figures/knn_roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["KNN"],
    "Train Accuracy": [train_accuracy],
    "Test Accuracy": [test_accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1-score": [f1],
    "ROC-AUC": [roc_auc],
    "Train-Test Gap": [train_test_gap]
})
