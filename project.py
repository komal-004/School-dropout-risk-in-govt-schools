# ============================
# 1. Import Libraries
# ============================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# ============================
# 2. Load Dataset
# ============================

data = pd.read_csv("C:/Users/004ko/Downloads/student_performance_with_dropout.csv")

print(data.head())

# ============================
# 3. Data Cleaning
# ============================

data = data.drop_duplicates()
data = data.dropna()

# ============================
# 4. Encode Categorical Column
# ============================

le = LabelEncoder()

categorical_cols = ['gender', 'school_type', 'parent_education', 'final_grade']

for col in categorical_cols:
    if col in data.columns:
        data[col] = le.fit_transform(data[col])

# ============================
# 5. Features and Target
# ============================

X = data.drop(["dropout", "overall_score"], axis=1) # Drop 'student_id' as it's an identifier
y = data["dropout"]

# ============================
# 6. Train-Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 7. EDA (Graph)
# ============================

y.value_counts().plot(kind='bar')
plt.title("Dropout Distribution")
plt.show()

# ============================
# 8. Logistic Regression
# ============================

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ============================
# 9. Decision Tree
# ============================

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# ============================
# 10. KNN
# ============================

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# ============================
# 11. Evaluation
# ============================

def evaluate(name, y_test, y_pred):
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

evaluate("Logistic Regression", y_test, y_pred_lr)
evaluate("Decision Tree", y_test, y_pred_dt)
evaluate("KNN", y_test, y_pred_knn)

# ============================
# 12. Best Model
# ============================

models = {
    "Logistic Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "KNN": y_pred_knn
}

best_model = ""
best_score = 0

for name, preds in models.items():
    score = f1_score(y_test, preds)
    if score > best_score:
        best_score = score
        best_model = name

print("\nBest Model:", best_model)
print("Best F1 Score:", best_score)