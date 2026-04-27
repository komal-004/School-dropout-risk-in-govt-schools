# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# 2. Load Dataset
data = pd.read_excel("C:/Users/004ko/Downloads/final_dataset_with_dropout_fixed_ids.xlsx")
print(data.head())

# 3. Encode Categorical Columns
le = LabelEncoder()
for col in ['gender', 'school_type', 'parent_education']:
    data[col] = le.fit_transform(data[col])
# Encode target
y = le.fit_transform(data['dropout_status'])
X = data.drop(['dropout_status', 'student_id'], axis=1)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(eval_metric='mlogloss')
}
trained_models = {}
results = {}

# 7. Training + Evaluation
for name, model in models.items():
    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    trained_models[name] = model
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = f1
    print("\n")
    print(name)
    print("Accuracy:", acc)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 8. Best Model
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\n")
print("Best Model:", best_model_name)
print("Best F1 Score:", results[best_model_name])

# 9. ROC Curve (Best Model Only)
if best_model_name in ["Logistic Regression", "KNN", "SVM"]:
    X_used = X_test_scaled
else:
    X_used = X_test

y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_prob = best_model.predict_proba(X_used)
class_names = ["No Risk", "High Risk", "Dropout"]
plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({best_model_name})")
plt.legend()
plt.show()