import pandas as pd
df = pd.read_csv("data/text_data/crime_data.csv")
print(df.head())
import pandas as pd
import pandas as pd

# Load dataset (limit rows for speed)
df = pd.read_csv("data/text_data/crime_data.csv", nrows=50000)

# Clean column names 
df.columns = df.columns.str.strip()

print("Columns:\n", df.columns)

# -------------------------------
# STEP 1: Convert DATE column
# -------------------------------
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# -------------------------------
# STEP 2: Remove invalid rows
# -------------------------------
df = df.dropna(subset=['DATE'])

# -------------------------------
# STEP 3: Extract time features
# -------------------------------
df['hour'] = df['DATE'].dt.hour
df['day'] = df['DATE'].dt.day
df['month'] = df['DATE'].dt.month
df['year'] = df['DATE'].dt.year

# -------------------------------
# STEP 4: Encode crime type
# -------------------------------
df['crime_type'] = df['PRIMARY_TYPE'].astype('category').cat.codes

# -------------------------------
# STEP 5: Keep useful columns
# -------------------------------
df = df[['DATE','hour','day','month','year','crime_type']]

# -------------------------------
# STEP 6: Show processed data
# -------------------------------
print("\nProcessed Data:\n", df.head())
print("\nDataset shape:", df.shape)
# -------------------------------
# STEP 8: Create time series
# -------------------------------

# Group crimes per month  
crime_trend = df.groupby(['year', 'month']).size()

print("\nCrime Trend:\n", crime_trend.head(10))
# Convert to numpy array
import numpy as np

data = crime_trend.values

print("\nRaw data:", data[:10])
# Normalize data
data = data / max(data)

print("\nNormalized data:", data[:10])
# -------------------------------
# STEP 9: Create sequences
# -------------------------------

X = []
y = []

sequence_length = 3

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

print("\nX shape:", X.shape)
print("y shape:", y.shape)
# -------------------------------
# STEP 10: LSTM MODEL
# -------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("\nTraining model...\n")

model.fit(X, y, epochs=10, verbose=1)
history = model.fit(X, y, epochs=10, verbose=1)
import matplotlib.pyplot as plt

# Loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
predictions = model.predict(X)

plt.figure()
plt.plot(y, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()
# -------------------------------
# BOTH GRAPHS TOGETHER
# -------------------------------

plt.figure(figsize=(10,5))

# Subplot 1 - Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.title("Training Loss")

# Subplot 2 - Prediction
plt.subplot(1,2,2)
plt.plot(y, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")

plt.tight_layout()
plt.show()
