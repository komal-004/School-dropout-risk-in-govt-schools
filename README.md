#  Student Dropout Risk Prediction in Government Schools

##  Project Overview

This project focuses on predicting the risk of student dropout in government schools using multiple machine learning models. The system analyzes student-related features and classifies them into categories such as **No Risk, High Risk, and Dropout**.

---

##  Objectives

* Identify students at risk of dropping out
* Compare multiple machine learning algorithms
* Select the best-performing model using evaluation metrics
* Visualize model performance using ROC curves

---

## 🛠️ Technologies & Libraries Used

* Python
* Pandas (Data Handling)
* Matplotlib (Visualization)
* Scikit-learn (ML Models & Evaluation)
* XGBoost (Advanced Boosting Algorithm)

---

##  Dataset

The dataset is loaded from an Excel file containing student information such as:

* Gender
* School Type
* Parent Education
* Other academic and demographic features

Target Variable:

* `dropout_status` (Encoded into 3 classes):

  * 0 → No Risk
  * 1 → High Risk
  * 2 → Dropout

---

##  Workflow

###  1. Data Preprocessing

* Label Encoding for categorical variables
* Feature selection
* Train-test split (80-20)
* Feature scaling using StandardScaler

---

###  2. Models Used

The project compares the following models:

* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)
* Random Forest
* Support Vector Machine (SVM)
* XGBoost

---

###  3. Model Evaluation

Each model is evaluated using:

* Accuracy
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

###  4. Best Model Selection

* The model with the **highest F1 Score** is selected as the best model.

---

###  5. ROC Curve Visualization

* ROC curves are plotted for all three classes:

  * No Risk
  * High Risk
  * Dropout
* AUC (Area Under Curve) is calculated for performance comparison

---

##  Results

* Multiple models were trained and evaluated
* The best model is selected automatically based on F1 Score
* ROC Curve visualization provides insight into classification performance

---

##  Project Structure

```
Project.py
results/
README.md
```

---

##  How to Run the Project

1. Clone the repository:

```
git clone https://github.com/komal-004/School-dropout-risk-in-govt-schools.git
```

2. Install required libraries:

```
pip install pandas matplotlib scikit-learn xgboost openpyxl
```

3. Update dataset path in `Project.py` if needed:

```
data = pd.read_excel("your_dataset_path.xlsx")
```

4. Run the script:

```
python Project.py
```

---

##  Output

* Model performance metrics printed in terminal
* Confusion matrices and classification reports
* ROC Curve graph displayed using matplotlib

---

##  Future Improvements

* Hyperparameter tuning for better accuracy
* Use deep learning models
* Deploy as a web application
* Add real-time data prediction

---

##  Author

Komal Jangid
