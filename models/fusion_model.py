import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# LOAD LSTM DATA
# -------------------------------


# Dummy small data for now 
X_lstm = np.random.rand(100, 3)   # simulate sequence features

# -------------------------------
# LOAD CNN FEATURES
# -------------------------------
cnn_features = np.load("data/video_data/cnn_features.npy")

# Match sizes
min_len = min(len(X_lstm), len(cnn_features))

X_lstm = X_lstm[:min_len]
cnn_features = cnn_features[:min_len]

# -------------------------------
# FUSION (COMBINE FEATURES)
# -------------------------------
X_combined = np.concatenate([X_lstm, cnn_features], axis=1)

# Dummy target
y = np.random.rand(min_len)

print("Combined shape:", X_combined.shape)

# -------------------------------
# SIMPLE FUSION MODEL
# -------------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_combined.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("Training fusion model...")

model.fit(X_combined, y, epochs=5)