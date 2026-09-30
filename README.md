Heart Failure 30-Day Readmission Prediction

"Python Version" (https://img.shields.io/badge/python-3.8%2B-blue)
"Machine Learning" (https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
"Status" (https://img.shields.io/badge/status-Complete-brightgreen)
"License" (https://img.shields.io/badge/license-MIT-green)

A clinical machine learning project designed to predict 30-day hospital readmission among heart failure patients using patient demographic, clinical, and medical features.

The project implements and compares three classical machine learning algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

The primary objective is to identify patients at higher risk of readmission before discharge, allowing healthcare providers to consider targeted follow-up and post-discharge care.

«Note: This project is intended for educational and research purposes. It is not a clinical diagnostic system and should not be used to make medical decisions.»

---

📌 Table of Contents

- "Project Overview" (#-project-overview)
- "Objectives" (#-objectives)
- "Dataset" (#-dataset)
- "Target Variable" (#-target-variable)
- "Class Distribution" (#-class-distribution)
- "Project Structure" (#-project-structure)
- "Technologies Used" (#-technologies-used)
- "Data Preprocessing" (#-data-preprocessing)
- "Machine Learning Models" (#-machine-learning-models)
- "Model Performance" (#-model-performance)
- "Confusion Matrix Results" (#-confusion-matrix-results)
- "Selected Model" (#-selected-model)
- "Key Technical Insights" (#-key-technical-insights)
- "How to Run the Project" (#-how-to-run-the-project)
- "Generated Outputs" (#-generated-outputs)
- "Future Improvements" (#-future-improvements)
- "Limitations" (#-limitations)
- "License" (#-license)

---

📌 Project Overview

Hospital readmission within 30 days is an important healthcare quality and operational metric. Predicting which heart failure patients may be readmitted can help healthcare organizations identify patients who may require additional monitoring and follow-up.

This project develops an end-to-end machine learning pipeline for predicting whether a patient will be readmitted within 30 days of discharge.

The workflow includes:

1. Data loading
2. Exploratory data analysis
3. Data cleaning
4. Feature preprocessing
5. Train-test splitting
6. Model training
7. Hyperparameter configuration
8. Model evaluation
9. Performance comparison
10. Visualization and output generation

Three classification algorithms are evaluated and compared using multiple metrics, with particular attention given to Recall and ROC-AUC.

---

🎯 Objectives

The main objectives of this project are:

- Predict 30-day heart failure readmission.
- Build a complete machine learning classification pipeline.
- Compare multiple classical classification algorithms.
- Prevent data leakage during preprocessing.
- Handle missing numerical and categorical values.
- Properly encode categorical variables.
- Scale numerical features where required.
- Evaluate models using multiple classification metrics.
- Give particular importance to Recall, because missing a high-risk patient can be clinically important.
- Compare training and testing performance to identify potential overfitting.
- Generate reusable model outputs and visualizations.

---

📊 Dataset

The dataset contains:

- 12,000 patient records
- 36 clinical features
- One binary target variable

Dataset Characteristics

Property| Value
Total Records| 12,000
Clinical Features| 36
Training Records| 9,600
Testing Records| 2,400
Target Type| Binary Classification
Readmitted Class| 1
Not Readmitted Class| 0

The dataset is stored inside the "Data/" directory.

---

🎯 Target Variable

The target variable is:

Readmitted_30_Days

It is a binary classification variable:

Value| Meaning
"0"| Not readmitted within 30 days
"1"| Readmitted within 30 days

The objective of the machine learning models is to predict this target variable using the available patient features.

---

📈 Class Distribution

The dataset contains a moderate class imbalance:

Class| Description| Records| Percentage
0| Not Readmitted| 8,400| 70%
1| Readmitted| 3,600| 30%
Total| | 12,000| 100%

Because the target is not perfectly balanced, accuracy alone is not sufficient for evaluating model performance.

Therefore, this project also evaluates:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

📂 Project Structure

Heart_Failure_Readmission_Project/
│
├── Data/
│ └── dataset_12000_records.csv
│
├── Outputs/
│ ├── confusion_matrix/
│ ├── plots/
│ ├── model_results/
│ └── other generated outputs
│
├── Decision_Tree.py
├── KNN.py
├── Logistic_Regression.py
│
├── README.md
├── Report.md
├── requirements.txt
└── LICENSE

File Description

File / Directory| Description
"Data/"| Contains the input dataset
"Outputs/"| Contains generated plots, figures, and results
"Logistic_Regression.py"| Logistic Regression model pipeline
"KNN.py"| K-Nearest Neighbors model pipeline
"Decision_Tree.py"| Decision Tree model pipeline
"Report.md"| Detailed project report
"requirements.txt"| Required Python packages
"README.md"| Project documentation
"LICENSE"| MIT License

«Note: The ".venv/" virtual environment should generally not be uploaded to GitHub. It is recommended to keep it locally and add ".venv/" to ".gitignore".»

---

🛠️ Technologies Used

The project is implemented using Python and the Scikit-learn machine learning ecosystem.

Programming Language

- Python 3.8+

Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Machine Learning Techniques

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree Classification
- Train-Test Split
- Feature Scaling
- One-Hot Encoding
- Missing Value Imputation
- Hyperparameter configuration
- ROC-AUC analysis
- Confusion Matrix analysis

---

🔄 Data Preprocessing

A preprocessing pipeline is used to prepare the dataset for machine learning.

1. Identifier Removal

The "Patient_ID" column is removed because it represents a patient identifier rather than a useful predictive feature.

Patient_ID → Removed

---

2. Numerical Features

Numerical features are processed using:

- Median imputation
- Standard scaling

The numerical pipeline follows:

Missing Values
↓
Median Imputation
↓
StandardScaler

"StandardScaler" transforms numerical variables so that features are placed on a comparable scale.

---

3. Categorical Features

Categorical variables are processed using:

- Most-frequent-value imputation
- One-hot encoding

The categorical pipeline follows:

Missing Values
↓
Most Frequent Imputation
↓
One-Hot Encoding

The encoder uses:

OneHotEncoder(handle_unknown="ignore")

This prevents errors when previously unseen categorical values appear in the test data.

---

🔐 Data Leakage Prevention

Data leakage is carefully avoided by fitting preprocessing transformations only on the training data.

The dataset is divided using an 80/20 stratified train-test split:

Complete Dataset
│
├── 80% → Training Set
│
└── 20% → Testing Set

This results in:

- 9,600 training records
- 2,400 testing records

Stratification is used to preserve the class distribution between training and testing datasets.

Preprocessing parameters such as imputation values and scaling statistics are learned from the training data and then applied to the test data.

---

🤖 Machine Learning Models

1. Logistic Regression

Logistic Regression is used as a linear classification model for predicting the probability of 30-day readmission.

File:

Logistic_Regression.py

The model provides:

- Binary classification
- Probability estimates
- Interpretable coefficients
- ROC-AUC evaluation

---

2. K-Nearest Neighbors

K-Nearest Neighbors classifies patients based on the similarity between their feature values and nearby training examples.

File:

KNN.py

The project uses:

k = 7

Feature scaling is particularly important for KNN because the algorithm relies on distance calculations.

---

3. Decision Tree

The Decision Tree classifier learns decision rules from the patient features.

File:

Decision_Tree.py

The tree is constrained using:

max_depth = 6
min_samples_leaf = 5

These parameters help limit excessive tree growth and reduce the possibility of overfitting.

---

🏆 Model Performance

The models were evaluated on the 2,400-record test set.

Model| Train Accuracy| Test Accuracy| Precision| Recall| F1-Score| ROC-AUC
Logistic Regression| 89.46%| 89.50%| 0.8524| 0.7861| 0.8179| 0.9562
KNN (k=7)| 88.33%| 84.75%| 0.8404| 0.6069| 0.7048| 0.8981
Decision Tree| 86.29%| 84.50%| 0.7862| 0.6639| 0.7199| 0.8950

---

📊 Model Comparison

Logistic Regression

Train Accuracy : 89.46%
Test Accuracy : 89.50%
Precision : 0.8524
Recall : 0.7861
F1-Score : 0.8179
ROC-AUC : 0.9562

The training and testing accuracy are very close, indicating that the model's performance is relatively consistent between the training and testing datasets.

---

KNN

Train Accuracy : 88.33%
Test Accuracy : 84.75%
Precision : 0.8404
Recall : 0.6069
F1-Score : 0.7048
ROC-AUC : 0.8981

The difference between training and testing accuracy indicates some reduction in generalization performance.

---

Decision Tree

Train Accuracy : 86.29%
Test Accuracy : 84.50%
Precision : 0.7862
Recall : 0.6639
F1-Score : 0.7199
ROC-AUC : 0.8950

The constrained tree provides relatively consistent training and testing performance.

---

🧮 Confusion Matrix Results

The confusion matrices are based on the 2,400-record test set.

Metric| Logistic Regression| KNN| Decision Tree
True Negatives (TN)| 1,582| 1,597| 1,550
False Positives (FP)| 98| 83| 130
False Negatives (FN)| 154| 283| 242
True Positives (TP)| 566| 437| 478

Logistic Regression

TN = 1582
FP = 98
FN = 154
TP = 566

KNN

TN = 1597
FP = 83
FN = 283
TP = 437

Decision Tree

TN = 1550
FP = 130
FN = 242
TP = 478

---

🎯 Selected Model

Based on the project's stated objective of prioritizing Recall and ROC-AUC, Logistic Regression was selected as the model for this project.

It achieved:

- Test Accuracy: 89.50%
- Precision: 0.8524
- Recall: 0.7861
- F1-Score: 0.8179
- ROC-AUC: 0.9562
- False Negatives: 154

The model also showed very similar training and testing accuracy:

Train Accuracy = 89.46%
Test Accuracy = 89.50%

This indicates that the model did not show a large train-test accuracy gap on this dataset.

---

🧠 Key Technical Insights

1. Why Recall Is Important

In a readmission-risk prediction problem, a false negative represents a patient who was predicted as not being at risk but was subsequently readmitted within 30 days.

A false positive, on the other hand, represents a patient predicted to be at risk who was not readmitted.

For this reason, Recall is an important metric when the objective is to identify as many potentially high-risk patients as possible.

Recall is calculated as:

Recall = TP / (TP + FN)

---

2. Importance of Feature Scaling

Feature scaling is particularly important for distance-based algorithms such as KNN.

For example, a feature with values in the thousands can dominate a feature whose values are between 0 and 1 if the data is not properly scaled.

Therefore, numerical features are standardized using:

StandardScaler()

---

3. One-Hot Encoding

Machine learning algorithms generally require numerical inputs.

Categorical variables are therefore converted into numerical representations using:

OneHotEncoder(handle_unknown="ignore")

---

4. Decision Tree Regularization

Decision trees can grow very deep and memorize training data.

To reduce this risk, the project uses:

max_depth=6
min_samples_leaf=5

These constraints limit tree complexity.

---

5. ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two classes across different classification thresholds.

A higher ROC-AUC indicates stronger ranking/discrimination performance on the evaluated dataset.

The Logistic Regression model achieved:

ROC-AUC = 0.9562

---

🚀 How to Run the Project

Step 1 — Clone the Repository

Clone the repository using Git:

git clone https://github.com/YOUR_USERNAME/Heart_Failure_Readmission_Project.git

Move into the project directory:

cd Heart_Failure_Readmission_Project

---

Step 2 — Create a Virtual Environment

On Windows:

python -m venv .venv

---

Step 3 — Activate the Virtual Environment

Windows Command Prompt

.venv\Scripts\activate

Windows PowerShell

.\.venv\Scripts\Activate.ps1

---

Step 4 — Install Dependencies

Run:

pip install -r requirements.txt

---

Step 5 — Run Logistic Regression

python Logistic_Regression.py

---

Step 6 — Run KNN

python KNN.py

---

Step 7 — Run Decision Tree

python Decision_Tree.py

Generated figures and model results will be stored in the "Outputs/" directory according to the configuration of each script.

---

📦 Requirements

The project dependencies are listed in:

requirements.txt

Typical dependencies include:

pandas
numpy
matplotlib
seaborn
scikit-learn

Install all dependencies with:

pip install -r requirements.txt

---

📁 Generated Outputs

The project can generate various machine learning outputs, including:

- Confusion matrices
- ROC curves
- Model performance comparisons
- Classification reports
- Prediction results
- Evaluation plots
- Other analysis figures

These outputs are stored in:

Outputs/

---

📈 Evaluation Metrics

The following metrics are used to evaluate the models.

Accuracy

Measures the proportion of correctly classified observations.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision

Measures how many predicted positive cases were actually positive.

Precision = TP / (TP + FP)

Recall

Measures how many actual positive cases were correctly identified.

Recall = TP / (TP + FN)

F1-Score

The harmonic mean of Precision and Recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

ROC-AUC

Measures the model's ability to distinguish between positive and negative classes across classification thresholds.

---

🔬 Future Improvements

Possible future improvements include:

- Testing additional classification algorithms.
- Hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
- Evaluating ensemble methods such as Random Forest and Gradient Boosting.
- Exploring XGBoost or other gradient boosting approaches.
- Applying class-weighting techniques.
- Investigating threshold optimization for improved recall.
- Performing cross-validation.
- Adding model explainability using SHAP or similar techniques.
- Performing feature importance analysis.
- Building an interactive Streamlit dashboard.
- Deploying the trained model through an API.
- Monitoring model performance on new data.
- Evaluating calibration of predicted probabilities.

---

⚠️ Limitations

This project has several limitations:

1. The dataset is a fixed dataset and may not represent all heart failure patient populations.
2. Model performance on this dataset does not guarantee the same performance on external datasets.
3. The models are intended for educational and research purposes.
4. Clinical deployment would require extensive external validation.
5. Patient data may contain biases or limitations that affect model predictions.
6. A machine learning prediction should not be treated as a medical diagnosis.
7. Additional clinical, ethical, privacy, and regulatory evaluation would be required before real-world deployment.

---

🏥 Clinical Use Disclaimer

This project is an educational machine learning project.

The predictions generated by the models should not be used as a substitute for professional medical judgment, diagnosis, or treatment decisions.

Any real-world clinical deployment would require appropriate validation, clinical oversight, privacy protections, regulatory review, and monitoring.

---

📄 License

This project is licensed under the MIT License.

See the "LICENSE" file for more information.

---

👨‍💻 Author

Debarun Kanjilal

Heart Failure 30-Day Readmission Prediction using Machine Learning.

---

⭐ Project Summary

This project demonstrates an end-to-end machine learning workflow for predicting 30-day hospital readmission among heart failure patients.

Three classification algorithms were implemented and evaluated:

Logistic Regression
↓
K-Nearest Neighbors
↓
Decision Tree

The models were compared using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix

Based on the project's evaluation criteria, Logistic Regression was selected for the final project model, achieving a test accuracy of 89.50%, Recall of 78.61%, F1-Score of 0.8179, and ROC-AUC of 0.9562 on the provided test set.

---

🔗 Repository

https://github.com/Debarun-Kanjilal/Heart_Failure_Readmission_Project
