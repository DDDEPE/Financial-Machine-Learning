import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
train = pd.read_excel("C:/Users/k5mic/OneDrive/바탕 화면/금융기계학습/Data/lendingclub_traindata.xlsx")
test = pd.read_excel("C:/Users/k5mic/OneDrive/바탕 화면/금융기계학습/Data/lendingclub_testdata.xlsx")

# 2. Feature / Target 설정
features = ['home_ownership', 'income', 'dti', 'fico']
target = 'loan_status'

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# 3. XGBClassifier 모델 설정 및 학습
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=0,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 4. 예측 확률 → AUC 계산
y_pred_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = auc(fpr, tpr)

print("XGBoost AUC:", auc_score)

# 5. ROC Curve 시각화
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend()
plt.show()
