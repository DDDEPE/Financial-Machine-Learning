# 금융기계학습 (Financial Machine Learning) – 문제 풀이

본 저장소는 대학원 수업 **금융기계학습(Financial Machine Learning)** 에서 다룬  
문제 풀이 및 머신러닝 모델 구현 코드를 정리한 레포지토리입니다.

수업에서 학습한 이론을 실제 금융 데이터에 적용하여  
모델 학습, 예측, 그리고 성능 평가(ROC Curve, AUC)를 수행하는 것을 목표로 합니다.

---

## 저장소 구조

각 Python 파일은 수업 중 제시된 문제 번호(Quiz 또는 과제)에 맞추어 작성되었습니다.
├── quiz_1_2.py 
├── quiz_3_4.py 
├── quiz_5_6.py 
├── quiz_7_8.py 
├── quiz_9_10.py 
├── quiz_11_12.py 
└── README.md

---

## 구현한 모델

본 저장소에서는 다음과 같은 머신러닝 모델을 구현하였습니다.

- 의사결정나무 (Decision Tree, Entropy 기준)
- 랜덤 포레스트 (Random Forest)
- 에이다부스트 (AdaBoost)
- 그래디언트 부스팅 (Gradient Boosting)
- XGBoost

모든 모델은 동일한 금융 데이터셋을 기반으로 학습되어  
모델 간 성능 비교가 가능하도록 구성하였습니다.

---

## 데이터셋

- **데이터**: Lending Club 대출 데이터 (수업 제공 자료)
- **타깃 변수**: `loan_status`
- **주요 설명 변수**
  - `home_ownership`
  - `income`
  - `dti`
  - `fico`

데이터는 학습용(train)과 테스트용(test)으로 구분되어 제공되었습니다.

> ※ 수업 규정 및 데이터 사용 정책에 따라 데이터 파일은 본 저장소에 포함하지 않았습니다.

---

## 성능 평가

모델 성능 평가는 다음 지표를 사용하였습니다.

- ROC Curve
- AUC (Area Under the Curve)

각 모델의 예측 확률을 기반으로 분류 성능을 평가하였습니다.

---

## 실행 환경

- Python 3.x
- pandas
- scikit-learn
- xgboost
- matplotlib

---

## 참고 사항

- 본 저장소는 **학습 및 복습 목적**으로 작성되었습니다.
- 하이퍼파라미터는 수업 조건 및 문제 요구사항을 반영하여 설정하였습니다.
- 실제 투자 또는 금융 의사결정을 위한 목적의 코드는 아닙니다.

---

## 작성자

신현민  
성균관대학교 핀테크융합전공 석사과정
