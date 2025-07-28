# Action and Grasp Classification Using Transformer

This repository contains the implementation of a Transformer-based model for **Action and Grasp Classification** using **Self-Attention** and **Multi-Head Attention** mechanisms. The model leverages the Transformer architecture to classify actions and grasps from time-series or sensor data (e.g., images, joint positions, etc.).

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Model Architecture](#model-architecture)
4. [Data Preparation](#data-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)


## Introduction

This project implements a Transformer model for classifying actions and grasps. The Transformer architecture utilizes **Self-Attention** and **Multi-Head Attention** layers to effectively model dependencies in sequential data. It is suitable for tasks involving time-series data or joint data from robotics, such as action recognition or grasp classification.

## Requirements

To run the code, ensure that the following libraries are installed:

* **PyTorch**: For model building and training.
* **NumPy**: For numerical operations.
* **Matplotlib, Seaborn**: For visualization.
* **scikit-learn**: For data preprocessing and evaluation metrics.

You can install the required dependencies using:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

## Model Architecture

The model consists of:

* **Embedding Layer**: This layer converts input features into higher-dimensional space for better representation.
* **Transformer Encoder Layer**: This is the core of the model, consisting of **Self-Attention** and **Multi-Head Attention** mechanisms.
* **Fully Connected Layer**: After passing through the Transformer layers, the output is processed by a fully connected layer to produce the final classification.

The architecture uses:

* **Input Dimensionality**: Input data has dimensions `(batch_size, seq_len, input_dim)`.
* **Output**: The model outputs class probabilities for different actions or grasps.

## Data Preparation

### Dataset

The model expects the input data in the form of time-series data, where each sample consists of multiple sequences of data points. Each data point is associated with labels, such as `Action Class` or `Grasp Class`.

* **Input Format**: `(batch_size, seq_len, input_dim)` where:

  * `batch_size`: Number of samples in a batch.
  * `seq_len`: Length of each sequence (e.g., number of time steps).
  * `input_dim`: The number of features per timestep (e.g., sensor readings or image features).

### Dataset Class

A custom `Dataset` class (using PyTorch) is used to load and preprocess the data. The dataset expects the data in `NumPy` or `Torch` tensor format.

```python
class ActionGraspDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

## Training

### Hyperparameters

* **`input_dim`**: Number of features in the input data.
* **`num_classes`**: Number of distinct classes (e.g., Action or Grasp classes).
* **`seq_len`**: Length of the input sequence.
* **`hidden_dim`**: Dimensionality of the hidden layers.
* **`num_heads`**: Number of attention heads in the transformer.
* **`num_layers`**: Number of transformer layers.

The model uses **CrossEntropyLoss** for classification and **Adam optimizer** for training.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

### Training Loop

The training loop uses the following process:

1. **Forward Pass**: The input sequence is passed through the transformer model.
2. **Loss Calculation**: The output is compared to the true labels using CrossEntropyLoss.
3. **Backward Pass**: Gradients are computed and used to update the model weights using the optimizer.

Example code for the training loop:

```python
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

## Evaluation

To evaluate the model, we compute the accuracy and the confusion matrix on a validation or test set.

Example evaluation code:

```python
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(labels, predicted)
        print(f"Accuracy: {accuracy:.4f}")
```

Additionally, we calculate the **classification report** and visualize the **confusion matrix**.

## Usage

### Example Input

You can use the trained model to predict the action or grasp for new input sequences.

```python
# Example usage for predicting a single sequence
input_sequence = np.random.rand(1, seq_len, input_dim)
input_tensor = torch.tensor(input_sequence, dtype=torch.float32)

# Get prediction
output = model(input_tensor)
_, predicted_class = torch.max(output, 1)
```

### Save and Load Model

You can save and load the model using PyTorch's `torch.save()` and `torch.load()` methods.

```python
# Save the model
torch.save(model.state_dict(), 'action_grasp_transformer.pth')

# Load the model
model.load_state_dict(torch.load('action_grasp_transformer.pth'))
model.eval()
```
