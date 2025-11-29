import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow import keras

# ================================
# 0. Seed 고정 (중요!!)
# ================================
RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ================================
# 1. Load Data (local project path)
# ================================
DATA_FILE = r"C:\Users\k5mic\OneDrive\바탕 화면\개발\Option_Data.csv"
df = pd.read_csv(DATA_FILE)

y = df[['Option Price with Noise','Option Price']]
X = df[['Spot price','Strike Price','Risk Free Rate','Volatility','Maturity','Dividend']]

# ================================
# 2. Data Split  (split도 seed 맞추기)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=RANDOM_SEED
)

# ================================
# 3. Scaling
# ================================
scaler = StandardScaler()
scaler.fit(X_train)

X_scaled_train = scaler.transform(X_train)
X_scaled_vals  = scaler.transform(X_val)
X_scaled_test  = scaler.transform(X_test)

y_train = np.asarray(y_train)
y_val   = np.asarray(y_val)
y_test  = np.asarray(y_test)

# ================================
# 4. Build NN Model (ReLU 버전)
# ================================
model_relu = keras.models.Sequential([
    Dense(20, activation="relu", input_shape=(6,)),
    Dense(20, activation="relu"),
    Dense(20, activation="relu"),
    Dense(1)
])

model_relu.summary()

# ================================
# 5. Compile (metric도 추가해서 mae 따로 저장)
# ================================
model_relu.compile(
    loss="mae",
    optimizer="adam",
    metrics=["mae"]
)

# ================================
# 6. Train (Epochs=10,000, batch_size 맞추기)
# ================================
history_relu = model_relu.fit(
    X_scaled_train,
    y_train[:, 0],
    epochs=10000,
    batch_size=128,
    verbose=1,
    validation_data=(X_scaled_vals, y_val[:, 0])
)

# ================================
# 7. Plot MAE Curve (same y-range)
# ================================
plt.figure(figsize=(10,5))
plt.plot(history_relu.history['mae'],     label='Training MAE')
plt.plot(history_relu.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.ylim(0.1, 0.21)
plt.title('Training vs Validation MAE (ReLU Activation, 10,000 epochs)')
plt.legend()
plt.grid(True)
plt.show()
