import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# %%
# 1. Dataset Creation (Synthetic Sine Wave)
def create_sine_wave(n_points=2000):
    time = np.linspace(0, 100, n_points)
    # Generate a sine wave and add some Gaussian noise
    series = np.sin(time) + np.random.normal(0, 0.1, n_points) 
    return series

series = create_sine_wave()

# %%
# 2. Data Windowing
# RNNs require data in (Samples, Time Steps, Features) format
def windowed_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

# %%
window_size = 50 # Look back at the previous 50 time steps
X, y = windowed_dataset(series, window_size)

# Reshaping dimensions: (Samples, Time Steps, Features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-Test Split (80% Training, 20% Testing)
split = int(0.8 * len(X))
x_train, x_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# %%
# 3. Model Architecture (SimpleRNN Example)
# Note: GRU or LSTM can also be used for better performance in complex sequences.
model = models.Sequential([
    layers.Input(shape=(window_size, 1)),
    # return_sequences=True is required when stacking RNN layers
    layers.SimpleRNN(64, activation='tanh', return_sequences=True), 
    layers.SimpleRNN(32),
    layers.Dense(1) # Predicting a single continuous value
])

# %%
# 4. Compilation
# Using Mean Squared Error (MSE) as it is a regression task
model.compile(optimizer='adam', loss='mse') 

# %%
# 5. Training
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=1
)

# %%
# 6. Prediction and Visualization
predictions = model.predict(x_test)

plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label="Actual Values", color='blue')
plt.plot(predictions[:200], label="RNN Predictions", color='red', linestyle='--')
plt.legend()
plt.title("Time Series Forecasting - Sine Wave")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.show()
