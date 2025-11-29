# ------------------------------------------------------------
# Option Pricing Neural Net (TensorFlow / Keras)
# 조건:
# (1) 3 hidden layers
# (2) 20 neurons per layer
# (3) Activation: Sigmoid
# (4) Loss: "MAE"
# (5) Optimizer: "Adam"
# (6) Epochs: 10,000
# (7) Train / Validation MAE vs Epoch 그래프 (y축 = 0.10~0.21)
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# 1. 파일 경로 / 기본 설정
# ------------------------------------------------------------
DATA_FILE = r"C:\Users\k5mic\OneDrive\바탕 화면\개발\Option_Data.csv"

FEATURE_COLS = [
    "Spot price",
    "Strike Price",
    "Risk Free Rate",
    "Volatility",
    "Maturity",
    "Dividend",
]
TARGET_COLS = ["Option Price with Noise", "Option Price"]

RANDOM_SEED = 100
BATCH_SIZE = 128
NUM_EPOCHS = 10_000

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------------------------------------
# 2. 데이터 로드 & Train/Val/Test 분리
# ------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

X = df[FEATURE_COLS].values.astype(np.float32)
y_all = df[TARGET_COLS].values.astype(np.float32)

# 타겟: Option Price with Noise
y = y_all[:, 0:1]

# 1) 전체 → Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# 2) Train → Train/Val
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=RANDOM_SEED
)

print("전체:", df.shape)
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ------------------------------------------------------------
# 3. 스케일링(StandardScaler)
# ------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)

# ------------------------------------------------------------
# 4. Keras 모델 (Sigmoid)
# ------------------------------------------------------------
input_dim = X_train_scaled.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(20, activation="sigmoid"),
    layers.Dense(20, activation="sigmoid"),
    layers.Dense(20, activation="sigmoid"),
    layers.Dense(1)
])

model.summary()

# ------------------------------------------------------------
# 5. 컴파일 (Loss = MAE, Optimizer = Adam)
# ------------------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="mae",
    metrics=["mae"]
)

# ------------------------------------------------------------
# 6. 학습 (Epoch = 10,000)
# ------------------------------------------------------------
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1   # 너무 길어서 숨김, 원하면 1로 변경
)

# 중간 확인
for e in [999, 4999, 9999]:
    print(f"Epoch {e+1}: Train MAE={history.history['mae'][e]:.5f}, "
          f"Val MAE={history.history['val_mae'][e]:.5f}")

# ------------------------------------------------------------
# 7. Train / Val MAE 그래프 (y축 = 0.10~0.21)
# ------------------------------------------------------------
epochs = range(1, NUM_EPOCHS + 1)
train_mae = history.history["mae"]
val_mae   = history.history["val_mae"]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_mae, label="Train MAE")
plt.plot(epochs, val_mae, label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Option Pricing NN (Sigmoid)\nTraining vs Validation MAE")
plt.ylim(0.10, 0.21)     # ★ y축 범위 고정 — 지글지글하게 보이도록
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
