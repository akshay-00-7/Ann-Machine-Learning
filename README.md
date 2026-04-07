# Neural Network Experiments — Deep Learning Assignment

This notebook is part of a deep learning assignment where I explored how different settings in a neural network affect its performance on an image classification task.

The dataset used is **IITM DL18 PA1** — it has 784 pixel features (like a flattened 28x28 image) and 10 output classes. I ran a bunch of experiments changing things like number of layers, neurons, optimizer, activation function, loss function, and batch size — one thing at a time — to see what actually makes a difference.

---

## Dataset

| Split | Samples |
|-------|---------|
| Train | 55,000 |
| Validation | 5,000 |
| Test | 10,000 |

Each sample has 784 features (feat0 to feat783). Pixel values were normalized to [0, 1] by dividing by 255 before training.

---

## What I experimented with

### 1. Number of neurons (1 hidden layer)
Activation: Sigmoid | Optimizer: Adam | Epochs: 30 | Batch size: 20

| Neurons | Train Acc | Val Acc |
|---------|-----------|---------|
| 50      | 93.24%    | 88.46%  |
| 100     | 94.75%    | 88.92%  |
| 200     | 95.72%    | 88.84%  |
| 300     | 96.05%    | 89.04%  |

More neurons helped with training accuracy but val accuracy didn't improve as much after 100 neurons — signs of slight overfitting with wider layers.

---

### 2. Two hidden layers

| Architecture  | Train Acc | Val Acc |
|---------------|-----------|---------|
| 50 → 50       | 93.03%    | 88.10%  |
| 100 → 100     | 94.48%    | 88.72%  |
| 200 → 200     | 95.75%    | 89.26%  |
| 300 → 300     | 96.11%    | 89.20%  |

Adding a second layer gave a small but consistent improvement, especially with 200-300 neurons.

---

### 3. Three hidden layers

| Architecture        | Train Acc | Val Acc |
|---------------------|-----------|---------|
| 50 → 50 → 50        | 93.19%    | 88.04%  |
| 100 → 100 → 100     | 94.28%    | 89.08%  |
| 200 → 200 → 200     | 95.33%    | 88.99%  |
| 300 → 300 → 300     | 95.49%    | 88.98%  |

Three layers didn't improve much over two. The 300-neuron version actually got slightly worse on val — probably a bit of overfitting starting to show.

---

### 4. Four hidden layers

| Architecture    | Train Acc | Val Acc |
|-----------------|-----------|---------|
| 50 × 4          | 92.43%    | 88.22%  |
| 100 × 4         | 93.78%    | 88.66%  |
| 200 × 4         | 94.77%    | 89.20%  |
| 300 × 4         | 94.80%    | 89.70%  |

4 layers with 300 neurons gave the best val accuracy (89.70%), even though train accuracy was lower than shallower networks. Deeper networks seem to generalize better here even if they're slower to train.

---

### 5. Optimizer comparison
Architecture: 300 → 300 → 300 | Activation: Sigmoid | Epochs: 30

| Optimizer        | Train Acc | Val Acc |
|------------------|-----------|---------|
| SGD (lr=0.01)    | 85.16%    | 84.80%  |
| SGD + Momentum   | 90.84%    | 88.26%  |
| NAG              | 91.35%    | 88.74%  |
| Adam (lr=0.001)  | 95.58%    | 89.46%  |

Plain SGD was way behind everyone else. Adam won clearly. NAG was slightly better than regular momentum but both were miles ahead of plain SGD.

---

### 6. Activation function
Architecture: 100 → 100 | Optimizer: Adam | Epochs: 30

| Activation | Train Acc | Val Acc |
|------------|-----------|---------|
| Sigmoid    | 94.66%    | 89.54%  |
| Tanh       | 93.70%    | 88.80%  |

Sigmoid did slightly better than tanh here, which was a bit unexpected. The gap isn't huge but it's consistent.

---

### 7. Loss function
Architecture: 100 → 100 | Activation: Sigmoid | Optimizer: Adam

| Loss Function  | Train Acc | Val Acc |
|----------------|-----------|---------|
| Cross Entropy  | 94.63%    | 88.96%  |
| MSE            | 9.79%     | 9.96%   |

MSE completely failed — basically random guessing the whole time. Cross entropy is the correct loss for classification. This experiment makes that point very clearly.

---

### 8. Batch size
Architecture: 100 → 100 | Activation: Sigmoid | Optimizer: Adam

| Batch Size | Train Acc | Val Acc |
|------------|-----------|---------|
| 1          | 90.19%    | 87.44%  |
| 20         | 94.56%    | 88.90%  |
| 100        | 93.56%    | 89.42%  |
| 1000       | 89.54%    | 88.20%  |

Batch size 1 was noisy and slow to converge. Batch 100 gave the best val accuracy. Very large batches like 1000 hurt generalization. The sweet spot seems to be somewhere between 20 and 100.

---

## How to run

This was run on Google Colab with a T4 GPU.

1. Upload `ASSIGMENT.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload the dataset zip to your Google Drive
3. Mount Drive and run all cells in order

For local use:
```bash
pip install tensorflow pandas numpy matplotlib jupyter
jupyter notebook ASSIGMENT.ipynb
```

Just update the file paths if running locally (remove the Drive mount cell).

---

## Files

```
ASSIGMENT.ipynb    main notebook with all experiments
train.csv          training data (55k samples)
val.csv            validation data (5k samples)
test.csv           test data, no labels (10k samples)
sample_sub.csv     submission format
```
