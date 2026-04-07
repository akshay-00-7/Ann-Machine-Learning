# 🧠 Deep Learning Assignment — Neural Network Experiments on Image Classification

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Sequential_API-red?logo=keras)](https://keras.io/)
[![Platform](https://img.shields.io/badge/Platform-Google_Colab-yellow?logo=google-colab)](https://colab.research.google.com/)
[![Dataset](https://img.shields.io/badge/Dataset-IITM_DL18_PA1-lightgrey)](.)

---

## 📌 Overview

This assignment systematically explores how different neural network design choices affect performance on a **10-class image classification task** (784 pixel features → 10 classes). Using the **IITM DL18 PA1** dataset (55,000 train / 5,000 val / 10,000 test samples), a series of controlled experiments are conducted by varying:

- Number of hidden layers (1 → 4)
- Number of neurons per layer (50, 100, 200, 300)
- Optimizer (SGD, Momentum, NAG, Adam)
- Activation function (Sigmoid vs Tanh)
- Loss function (Cross Entropy vs MSE)
- Batch size (1, 20, 100, 1000)

---

## 📂 Project Structure

```
📁 project/
│
├── ASSIGMENT.ipynb          # Main Jupyter notebook with all experiments
├── train.csv                # Training data (55,000 samples, 784 features + label)
├── val.csv                  # Validation data (5,000 samples)
├── test.csv                 # Test data (10,000 samples, no labels)
├── sample_sub.csv           # Sample submission format
└── README.md                # This file
```

---

## 📊 Dataset

| Split      | Samples | Features       |
|------------|---------|----------------|
| Train      | 55,000  | feat0–feat783 + label |
| Validation | 5,000   | feat0–feat783 + label |
| Test       | 10,000  | feat0–feat783 (no label) |

- Input: 784 pixel features (28×28 flattened image)
- Output: 10 classes (digits 0–9)
- **Preprocessing:** Pixel values normalized to `[0, 1]` by dividing by 255

---

## 🏗️ Experiments

### ✅ Baseline Model

A baseline model was first built with:
- Architecture: `784 → 300 → 300 → 10`
- Activation: `tanh`
- Optimizer: `Adam`
- Loss: `Sparse Categorical Cross Entropy`
- Epochs: 20, Batch Size: 20

---

### 🔬 Experiment 1 — Hidden Layer Width (1 Hidden Layer)

Fixed: 1 hidden layer, Sigmoid activation, Adam optimizer, 30 epochs

| Neurons | Train Accuracy | Val Accuracy |
|---------|---------------|--------------|
| 50      | ~             | ~            |
| 100     | ~             | ~            |
| 200     | ~             | ~            |
| 300     | ~             | ~            |

> 📈 Plots: Training Loss vs Epoch & Validation Loss vs Epoch

---

### 🔬 Experiment 2 — Hidden Layer Width (2 Hidden Layers)

Fixed: 2 hidden layers, Sigmoid activation, Adam optimizer, 30 epochs

| Architecture  | Train Accuracy | Val Accuracy |
|---------------|---------------|--------------|
| 50 → 50       | ~             | ~            |
| 100 → 100     | ~             | ~            |
| 200 → 200     | ~             | ~            |
| 300 → 300     | ~             | ~            |

---

### 🔬 Experiment 3 — Hidden Layer Width (3 Hidden Layers)

Fixed: 3 hidden layers, Sigmoid activation, Adam optimizer, 30 epochs

| Architecture        | Train Accuracy | Val Accuracy |
|---------------------|---------------|--------------|
| 50 → 50 → 50        | ~             | ~            |
| 100 → 100 → 100     | ~             | ~            |
| 200 → 200 → 200     | ~             | ~            |
| 300 → 300 → 300     | ~             | ~            |

---

### 🔬 Experiment 4 — Hidden Layer Width (4 Hidden Layers)

Fixed: 4 hidden layers, Sigmoid activation, Adam optimizer, 30 epochs

| Architecture              | Train Accuracy | Val Accuracy |
|---------------------------|---------------|--------------|
| 50 × 4                    | ~             | ~            |
| 100 × 4                   | ~             | ~            |
| 200 × 4                   | ~             | ~            |
| 300 × 4                   | ~             | ~            |

---

### 🔬 Experiment 5 — Optimizer Comparison

Fixed: 3 hidden layers (300→300→300), Sigmoid, 30 epochs, Batch Size 20

| Optimizer  | Train Accuracy | Val Accuracy |
|------------|---------------|--------------|
| SGD (lr=0.01)            | ~ | ~ |
| SGD + Momentum (0.9)     | ~ | ~ |
| NAG (Nesterov=True)      | ~ | ~ |
| Adam (lr=0.001)          | ~ | ~ |

---

### 🔬 Experiment 6 — Activation Function Comparison

Fixed: 2 hidden layers (100→100), Adam optimizer, 30 epochs

| Activation | Train Accuracy | Val Accuracy |
|------------|---------------|--------------|
| Sigmoid    | ~             | ~            |
| Tanh       | ~             | ~            |

---

### 🔬 Experiment 7 — Loss Function Comparison

Fixed: 2 hidden layers (100→100), Sigmoid, Adam, 30 epochs

| Loss Function           | Train Accuracy | Val Accuracy |
|-------------------------|---------------|--------------|
| Sparse Cross Entropy    | ~             | ~            |
| Mean Squared Error (MSE)| ~             | ~            |

---

### 🔬 Experiment 8 — Batch Size Comparison

Fixed: 2 hidden layers (100→100), Sigmoid, Adam, 30 epochs

| Batch Size | Train Accuracy | Val Accuracy |
|------------|---------------|--------------|
| 1          | ~             | ~            |
| 20         | ~             | ~            |
| 100        | ~             | ~            |
| 1000       | ~             | ~            |

---

## 🛠️ Tech Stack

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| Python 3.x       | Programming language             |
| TensorFlow 2.x   | Deep learning framework          |
| Keras            | Model building (Sequential API)  |
| Pandas           | Data loading & manipulation      |
| NumPy            | Numerical operations             |
| Matplotlib       | Plotting loss/accuracy curves    |
| Google Colab     | GPU-accelerated training (T4)    |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

1. Upload `ASSIGMENT.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload the dataset zip (`iitm-dl18-pa1.zip`) to your Google Drive
3. Mount Drive and run all cells in order

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install tensorflow pandas numpy matplotlib

# Launch Jupyter Notebook
jupyter notebook ASSIGMENT.ipynb
```

> ⚠️ **Note:** Remove the Google Drive mount cell if running locally. Update dataset paths accordingly.

---

## 📋 Requirements

```
tensorflow>=2.0
pandas
numpy
matplotlib
jupyter
```

---

## 📈 Key Observations

- **More neurons** generally improve accuracy but increase training time and risk overfitting.
- **Deeper networks** (more layers) can learn more complex patterns but may suffer from vanishing gradients with sigmoid activation.
- **Adam optimizer** converges faster and more reliably than vanilla SGD.
- **Tanh** tends to outperform **Sigmoid** due to zero-centered outputs.
- **Cross Entropy loss** is significantly better suited for classification than MSE.
- **Smaller batch sizes** (e.g., 20) often generalize better but are slower to train; very large batches (e.g., 1000) converge faster per epoch but may generalize worse.

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com

---

## 📜 License

This project is for academic/educational purposes as part of a Deep Learning course assignment.
