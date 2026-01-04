import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# 3. 범주형 변수(one-hot) 처리
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# train/test 컬럼 동일하게 맞추기
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 4. Random Forest 모델 생성
rf = RandomForestClassifier(
    n_estimators=300,
    criterion='entropy',
    max_depth=4,
    random_state=0,
    min_samples_split=500,
    min_samples_leaf=50,
    max_samples=0.7
)

# 5. 모델 학습
rf.fit(X_train, y_train)

# 6. ROC Curve & AUC 계산
y_pred_proba = rf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 7. ROC Curve Plot
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"Random Forest AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.show()

print("AUC:", roc_auc)
