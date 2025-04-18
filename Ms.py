import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# Read data
data_path = r"T.xlsx"  # Updated path to the dataset
df = pd.read_excel(data_path)

# Separate features and target variable
y = df['Ms (K)'].values
excluded = ['Ms (K)']
X = df.drop(excluded, axis=1).values

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define ResNet model
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.layer1 = nn.Linear(in_channels, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, out_channels)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.relu(out)
        out = self.layer2(out)
        out = torch.relu(out)
        out = self.layer3(out)
        out = self.dropout(out)
        return out


class ResNetModel(nn.Module):
    def __init__(self, input_size):
        super(ResNetModel, self).__init__()
        self.block1 = ResNetBlock(input_size, 8)
        self.block2 = ResNetBlock(8, 8)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.fc(out)
        return out


# Instantiate the model
model = ResNetModel(X_train.shape[1])

# Use Adam optimizer and cosine annealing LR scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=10000)
criterion = nn.MSELoss()


# Training and validation function
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=10000, patience=2000):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    val_loss_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Adjust learning rate
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.flatten(), labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Check early stopping condition
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            val_loss_no_improve = 0
        else:
            val_loss_no_improve += 1

        if val_loss_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}")

    # Load best model
    model.load_state_dict(best_model_wts)
    return model


# Train the model
trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion)

# Predict using the trained model
model.eval()
with torch.no_grad():
    # Predict for training set
    y_train_pred = model(X_train_tensor).numpy().flatten()

    # Predict for validation set
    y_val_pred = model(X_val_tensor).numpy().flatten()

# Calculate R2 and RMSE for both training and validation sets
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"Train R2: {r2_train}")
print(f"Train RMSE: {rmse_train}")
print(f"Validation R2: {r2_val}")
print(f"Validation RMSE: {rmse_val}")

# True vs. predicted scatter plot (training vs validation sets)
plt.figure(figsize=(8, 6))

# Plot for training set
plt.scatter(y_train, y_train_pred, label="Train", color='blue', alpha=0.6)

# Plot for validation set
plt.scatter(y_val, y_val_pred, label="Validation", color='red', alpha=0.6)

# Plot the perfect prediction line
plt.plot([min(y), max(y)], [min(y), max(y)], label="Perfect Prediction", color='black', linestyle='--')

plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted (Train vs Validation)")
plt.legend()
plt.show()

# Save the model
torch.save(trained_model.state_dict(), 'T.pth')
