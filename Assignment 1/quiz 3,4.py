import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# -----------------------------
# 1. 데이터 불러오기 (로컬 경로)
# -----------------------------
train = pd.read_excel("C:/Users/k5mic/OneDrive/바탕 화면/금융기계학습/Data/lendingclub_traindata.xlsx")
test = pd.read_excel("C:/Users/k5mic/OneDrive/바탕 화면/금융기계학습/Data/lendingclub_testdata.xlsx")

# -----------------------------
# 2. Feature / Target 설정
# -----------------------------
features = ['home_ownership', 'income', 'dti', 'fico']
target = 'loan_status'

train_enc = pd.get_dummies(train[features], drop_first=True)
test_enc = pd.get_dummies(test[features], drop_first=True)

train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)

y_train = train[target]
y_test = test[target]

# -----------------------------
# 3. Random Forest 모델 설정
# -----------------------------
rf = RandomForestClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=0
)

rf.fit(train_enc, y_train)

# -----------------------------
# 4. 예측 및 ROC/AUC 계산
# -----------------------------
probs = rf.predict_proba(test_enc)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

print(f"AUC Score: {roc_auc:.4f}")

# -----------------------------
# 5. ROC Curve 그리기
# -----------------------------
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Random Forest AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
