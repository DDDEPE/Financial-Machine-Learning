# ------------------------------------------------------------
# Implied Volatility Neural Network (3×20 Sigmoid)
# - Loss: MSE
# - Optimizer: Adam
# - Epochs: 10,000
# - Output: Train / Validation MSE vs Epoch
# ------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Dense
from tensorflow import keras

# ======================================
# 1. Load Data
#   (이 파이썬 파일과 같은 폴더에
#    Implied_Volatility_Data_vFinal.csv 가 있다고 가정)
# ======================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Implied_Volatility_Data_vFinal.csv")

raw = pd.read_csv(data_path)
print("Data shape:", raw.shape)

# ======================================
# 2. Feature Engineering  (수업 예제 그대로)
# ======================================
raw["x1"] = raw["SPX Return"] / np.sqrt(raw["Time to Maturity in Year"])
raw["x2"] = raw["x1"] * raw["Delta"]
raw["x3"] = raw["x2"] * raw["Delta"]

# 총 6개 입력변수
X_imp = raw[[
    "x1", "x2", "x3",
    "SPX Return",
    "Time to Maturity in Year",
    "Delta"
]]

y_imp = raw["Implied Volatility Change"]

# ======================================
# 3. Train / Validation / Test Split
#    (Train 60%, Val 20%, Test 20%)
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y_imp, test_size=0.2, random_state=100
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=100
)

# ======================================
# 4. Scaling (입력만 표준화)
# ======================================
scaler = StandardScaler()
scaler.fit(X_train)

Xs_train = scaler.transform(X_train)
Xs_val   = scaler.transform(X_val)
Xs_test  = scaler.transform(X_test)

y_train = np.asarray(y_train)
y_val   = np.asarray(y_val)
y_test  = np.asarray(y_test)

# ======================================
# 5. Neural Net Model (3 x 20, Sigmoid)
# ======================================
imvol_model = keras.models.Sequential([
    Dense(20, activation="sigmoid", input_shape=(6,)),
    Dense(20, activation="sigmoid"),
    Dense(20, activation="sigmoid"),
    Dense(1)  # 회귀 출력층 (linear)
])

imvol_model.summary()

# ======================================
# 6. Compile (Loss=MSE, Optimizer=Adam)
# ======================================
imvol_model.compile(loss="mse", optimizer="adam")

# ======================================
# 7. Train (Epochs = 10,000)
# ======================================
imvol_history = imvol_model.fit(
    Xs_train,
    y_train,
    epochs=10000,
    batch_size=128,
    verbose=1,                     # 학습로그 보고 싶으면 1, 안보고 싶으면 0
    validation_data=(Xs_val, y_val)
)

# ======================================
# 8. Plot Result (수업 예제 스타일)
# ======================================
plt.figure(figsize=(10, 5))
plt.plot(imvol_history.history["loss"],     label="Training Set")
plt.plot(imvol_history.history["val_loss"], label="Validation Set")
plt.grid(True)

# 예제와 비슷한 y축 범위 설정 (10^-4 수준)
plt.gca().set_ylim(0.00004, 0.00015)

plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Implied Volatility NN: Training / Validation MSE vs Epoch\n(3×20 Sigmoid, Loss=MSE, Adam)")
plt.legend()

out_png = os.path.join(BASE_DIR, "NN_MSE_ImpliedVol.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.show()

print(f"그래프가 '{out_png}' 파일로 저장되었습니다.")
