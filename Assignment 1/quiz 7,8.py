import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. 데이터 불러오기
# ----------------------------------------
train = pd.read_excel("C:/Users/k5mic/OneDrive/바탕 화면/금융기계학습/Data/lendingclub_traindata.xlsx")
test = pd.read_excel("C:/Users/k5mic/OneDrive/바탕 화면/금융기계학습/Data/lendingclub_testdata.xlsx")

# ----------------------------------------
# 2. Feature / Target 설정
# ----------------------------------------
features = ['home_ownership', 'income', 'dti', 'fico']
target = 'loan_status'

X_train = train[features].copy()
y_train = train[target]

X_test = test[features].copy()
y_test = test[target]

# ----------------------------------------
# 3. 범주형(home_ownership) Label Encoding
# ----------------------------------------
le = LabelEncoder()
X_train['home_ownership'] = le.fit_transform(X_train['home_ownership'])
X_test['home_ownership'] = le.transform(X_test['home_ownership'])

# ----------------------------------------
# 4. AdaBoost 모델 정의 및 학습
# ----------------------------------------
model = AdaBoostClassifier(
    n_estimators=200,
    random_state=0
)

model.fit(X_train, y_train)

# ----------------------------------------
# 5. 예측 확률 계산
# ----------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------------------
# 6. ROC Curve 및 AUC 계산
# ----------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)

# ----------------------------------------
# 7. ROC Curve 그리기
# ----------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AdaBoost (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')

plt.title("ROC Curve - AdaBoost Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
