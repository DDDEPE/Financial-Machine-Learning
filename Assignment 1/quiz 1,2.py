import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# -----------------------------
# 1. 데이터 로드
# -----------------------------
train = pd.read_excel("C:/Users/k5mic/OneDrive/바탕 화면/금융기계학습/Data/lendingclub_traindata.xlsx")
test = pd.read_excel("C:/Users/k5mic/OneDrive/바탕 화면/금융기계학습/Data/lendingclub_testdata.xlsx")

# 필요한 변수 선택
features = ["home_ownership", "income", "dti", "fico"]
target = "loan_status"

# One-hot encoding (home_ownership: 범주형)
train_enc = pd.get_dummies(train[features], drop_first=True)
test_enc = pd.get_dummies(test[features], drop_first=True)

# train/test 간 컬럼 불일치 해결
missing_cols = set(train_enc.columns) - set(test_enc.columns)
for c in missing_cols:
    test_enc[c] = 0
test_enc = test_enc[train_enc.columns]

X_train = train_enc
y_train = train[target]

X_test = test_enc
y_test = test[target]

# -----------------------------
# 2. Decision Tree 모델 설정
# -----------------------------
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=500,
    min_samples_leaf=50,
    random_state=0
)

# -----------------------------
# 3. 모델 학습
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 4. 예측 및 AUC 계산
# -----------------------------
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_value = roc_auc_score(y_test, y_pred_proba)

print("AUC Score:", auc_value)

# -----------------------------
# 5. ROC Curve Plot
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'AUC = {auc_value:.4f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Decision Tree)")
plt.legend()
plt.grid()
plt.show()
