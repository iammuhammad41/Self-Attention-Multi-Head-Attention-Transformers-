import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define Transformer-based model
class ActionGraspTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=6, hidden_dim=512, dropout=0.1):
        super(ActionGraspTransformer, self).__init__()

        # Embedding layer (if input is time-series or image-like data)
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)

        # Pass through embedding layer
        x = self.embedding(x)  # Shape: (batch_size, seq_len, hidden_dim)

        # Reshape to match the shape expected by transformer (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)

        # Use the output of the last token for classification (can also use pooling)
        x = x[-1, :, :]  # Shape: (batch_size, hidden_dim)

        # Classification layer
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x


# Dataset class (assuming the data is in numpy arrays or tensors)
class ActionGraspDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (np.array or torch.Tensor): Input features of shape (num_samples, seq_len, input_dim)
            labels (np.array or torch.Tensor): Corresponding labels of shape (num_samples,)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Model Hyperparameters
input_dim = 256  # Example input dimension (e.g., 256 features per timestep)
num_classes = 3  # For example: 0 = action1, 1 = action2, 2 = action3
seq_len = 50  # Number of timesteps (sequence length)

# Initialize the model
model = ActionGraspTransformer(input_dim=input_dim, num_classes=num_classes)

# Optimizer and Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Example data (num_samples x seq_len x input_dim)
# For simplicity, let's assume random data for training/testing
num_samples = 1000
data = np.random.rand(num_samples, seq_len, input_dim)  # Random data
labels = np.random.randint(0, num_classes, num_samples)  # Random labels

# Create Dataset and DataLoader
train_dataset = ActionGraspDataset(data, labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)
        loss.backward()

        # Update weights
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)

    # Print statistics after every epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Evaluation (use your validation or test dataset here)
model.eval()
# Assume you have a validation or test dataset
# test_data, test_labels = ...
# test_dataset = ActionGraspDataset(test_data, test_labels)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# predictions = []
# targets = []
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         predictions.append(predicted)
#         targets.append(labels)
#     predictions = torch.cat(predictions)
#     targets = torch.cat(targets)
#     print("Accuracy:", accuracy_score(targets.numpy(), predictions.numpy()))
