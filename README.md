# Deep Learning with RNN & LSTM

This repository explores sequence modeling using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. It contains two distinct implementations demonstrating regression and classification tasks.

## 📁 Project Structure
1. **Time Series Prediction (SimpleRNN):** Predicting the next value in a noisy sine wave.
2. **Fashion-MNIST Classification (LSTM):** Classifying clothing items by treating image rows as sequences.

---

## 📌 1. Time Series Prediction (SimpleRNN)
This project uses a **SimpleRNN** to learn patterns in a periodic signal.
- **Goal:** Predict future values based on the past 50 time steps.
- **Key Techniques:** Data windowing, Many-to-One architecture.
- **File:** `sine_wave_rnn.py`

## 📌 2. Image Classification with LSTM (Fashion-MNIST)
This project treats 28x28 images as a sequence of 28 rows to perform classification.
- **Goal:** Categorize images into 10 clothing types.
- **Key Techniques:** LSTM layers, Dropout for regularization, Softmax activation.
- **File:** `fashion_mnist_lstm.py`

---

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
