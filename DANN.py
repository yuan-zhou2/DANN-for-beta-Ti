import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from calculate_params import Parameter_Calculate
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
import itertools

# -----------------------------
# Reproducibility settings
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Check for available GPU
device = torch.device('cpu')
print(f"Using device: {device}")

# Initialize parameter calculator
parameter_calculator = Parameter_Calculate()


def calculate_params(df):
    df = parameter_calculator.calculate_count(df)
    df = parameter_calculator.calculate_count_num(df)
    return df


# Define gradient reversal layer
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# Define DANN model
class DANN(nn.Module):
    def __init__(self, input_size, feature_node, regressor_node1, regressor_node2, dropout_rates=(0.2, 0.2, 0.2)):
        super(DANN, self).__init__()
        dropout_feature, dropout_regressor, dropout_domain = dropout_rates

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, feature_node),
            nn.PReLU(init=0.165),
            nn.Dropout(dropout_feature),
            nn.Linear(feature_node, 12),
            nn.Sigmoid()
        )

        self.regressor = nn.Sequential(
            nn.Linear(12, regressor_node1),
            nn.PReLU(init=0.165),
            nn.Dropout(dropout_regressor),
            nn.Linear(regressor_node1, regressor_node2),
            nn.PReLU(init=0.165),
            nn.Linear(regressor_node2, 1),
            nn.PReLU(init=0.165),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(12, 15),
            nn.PReLU(init=0.165),
            nn.Dropout(dropout_domain),
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_data, alpha=0.0):
        features = self.feature_extractor(input_data)
        reverse_features = ReverseLayerF.apply(features, alpha)
        regression_output = self.regressor(features)
        domain_output = self.domain_classifier(reverse_features)
        return regression_output, domain_output, features


# Data loading and preprocessing
def load_and_preprocess_data(data_path):
    df_source = pd.read_excel(data_path)
    df_source = calculate_params(df_source)

    y = df_source['E (GPa)'].values
    excluded = ['E (GPa)']
    X = df_source.drop(excluded, axis=1).values

    # Split source domain (training set) and target domain (validation set)
    X_source, X_target, y_source, y_target = train_test_split(
        X, y, test_size=0.15, random_state=18
    )

    # Normalize features
    scaler_X = MinMaxScaler()
    X_source = scaler_X.fit_transform(X_source)
    X_target = scaler_X.transform(X_target)

    # Normalize target variable
    scaler_y = MinMaxScaler()
    y_source = scaler_y.fit_transform(y_source.reshape(-1, 1))
    y_target = scaler_y.transform(y_target.reshape(-1, 1))

    return X_source, X_target, y_source, y_target, scaler_y


# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, y=None, domain_label=0):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
        self.domain_label = torch.tensor(domain_label, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx], self.domain_label
        else:
            return self.X[idx], self.domain_label


# Training and validation function with proper early stopping
def train_model(model, source_loader, target_loader, optimizer, criterion_reg, criterion_domain,
                num_epochs=80000, alpha=1.0, loss_threshold=0.006, consecutive_epochs=100):
    model.to(device)
    best_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    low_loss_epochs = 0
    patience = 2000
    patience_reduced = False

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for epoch in range(num_epochs):
        model.train()
        total_reg_loss = 0.0
        total_domain_loss = 0.0

        for _ in range(len(source_loader)):
            try:
                X_source, y_source, domain_label_source = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                X_source, y_source, domain_label_source = next(source_iter)

            try:
                X_target, _, domain_label_target = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                X_target, _, domain_label_target = next(target_iter)

            X_source = X_source.to(device)
            y_source = y_source.to(device)
            domain_label_source = domain_label_source.to(device)
            X_target = X_target.to(device)
            domain_label_target = domain_label_target.to(device)

            optimizer.zero_grad()

            reg_out, domain_out_source, _ = model(X_source, alpha=alpha)
            _, domain_out_target, _ = model(X_target, alpha=alpha)

            loss_reg = criterion_reg(reg_out, y_source)
            domain_out = torch.cat((domain_out_source, domain_out_target), 0)
            domain_labels = torch.cat((domain_label_source, domain_label_target), 0).unsqueeze(1)
            loss_domain = criterion_domain(domain_out, domain_labels)

            loss = loss_reg + loss_domain
            loss.backward()
            optimizer.step()

            total_reg_loss += loss_reg.item() * X_source.size(0)
            total_domain_loss += loss_domain.item() * (X_source.size(0) + X_target.size(0))

        avg_reg_loss = total_reg_loss / len(source_loader.dataset)
        avg_domain_loss = total_domain_loss / (len(source_loader.dataset) + len(target_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val, _ in target_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                reg_out, _, _ = model(X_val, alpha=alpha)
                loss_reg = criterion_reg(reg_out, y_val)
                val_loss += loss_reg.item() * X_val.size(0)

        val_loss /= len(target_loader.dataset)

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train regression loss: {avg_reg_loss:.6f}, '
                  f'Train domain classification loss: {avg_domain_loss:.6f}, '
                  f'Validation loss: {val_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_loss < loss_threshold:
            low_loss_epochs += 1
            print(f'Validation loss below {loss_threshold}, for {low_loss_epochs} consecutive epochs.')
        else:
            low_loss_epochs = 0

        if low_loss_epochs >= consecutive_epochs and not patience_reduced:
            patience = 500
            patience_reduced = True
            print(f'Validation loss has been below {loss_threshold} for {consecutive_epochs} consecutive epochs. '
                  f'Early stopping patience reduced to {patience} epochs.')

        if epochs_no_improve >= patience:
            print(f'\nEarly stopping: validation loss did not improve for {patience} epochs.')
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    return model


# Evaluation function
def evaluate_model(model, loader, scaler_y):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device)
            reg_out, _, _ = model(X)
            preds.extend(reg_out.cpu().numpy())
            targets.extend(y.numpy())

    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    return preds, targets, rmse, r2


# Plot predictions
def plot_predictions(train_preds, train_targets, val_preds, val_targets, rmse_train, r2_train,
                     rmse_val, r2_val, alpha):
    plt.rcParams["font.family"] = "Times New Roman"
    rcParams.update({'font.size': 18})
    plt.figure(figsize=(8, 6))
    plt.scatter(train_targets, train_preds, alpha=0.6, label='Train Predictions', color='blue')
    plt.scatter(val_targets, val_preds, alpha=0.6, label='Val Predictions', color='orange')
    min_val = min(train_targets.min(), val_targets.min())
    max_val = max(train_targets.max(), val_targets.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual E (GPa)')
    plt.ylabel('Predicted E (GPa)')
    plt.title('Predictions vs Actual (Train & Val)')
    plt.text(0.05, 0.95, f'Val RMSE: {rmse_val:.4f}\nVal R²: {r2_val:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# Plot t-SNE visualization
def plot_tsne(model, source_loader, target_loader):
    model.eval()
    source_features = []
    target_features = []
    with torch.no_grad():
        for X, _, _ in source_loader:
            X = X.to(device)
            features = model.feature_extractor(X)
            source_features.append(features.cpu().numpy())
        for X, _, _ in target_loader:
            X = X.to(device)
            features = model.feature_extractor(X)
            target_features.append(features.cpu().numpy())

    source_features = np.concatenate(source_features, axis=0)
    target_features = np.concatenate(target_features, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    combined_features = np.vstack((source_features, target_features))
    tsne_result = tsne.fit_transform(combined_features)

    source_tsne = tsne_result[:len(source_features)]
    target_tsne = tsne_result[len(source_features):]

    plt.figure(figsize=(8, 6))
    plt.scatter(source_tsne[:, 0], source_tsne[:, 1], label='Source Domain (Train)', alpha=0.6, color='blue')
    plt.scatter(target_tsne[:, 0], target_tsne[:, 1], label='Target Domain (Val)', alpha=0.6, color='red')
    plt.xlabel('t-SNE Dimension1')
    plt.ylabel('t-SNE Dimension2')
    plt.title('t-SNE of Source and Target Domain Features After Adaptation')
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()


# Main function
def main():
    data_path = r"data.xlsx"
    feature_node = 18
    regressor_node1 = 11
    regressor_node2 = 16
    dropout_rates = (0.1, 0.1, 0.1)
    learning_rate = 0.00023
    num_epochs = 80000
    loss_threshold = 0.0046
    consecutive_epochs = 200
    alpha = 0.2

    X_source, X_target, y_source, y_target, scaler_y = load_and_preprocess_data(data_path)

    source_dataset = CustomDataset(X_source, y_source, domain_label=0)
    target_dataset = CustomDataset(X_target, y_target, domain_label=1)

    # Add same random seed generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(SEED)

    source_loader = DataLoader(source_dataset, batch_size=len(source_dataset), shuffle=True, generator=generator)
    target_loader = DataLoader(target_dataset, batch_size=len(source_dataset), shuffle=True, generator=generator)

    input_size = X_source.shape[1]
    model = DANN(input_size, feature_node, regressor_node1, regressor_node2, dropout_rates=dropout_rates)

    criterion_reg = nn.MSELoss()
    criterion_domain = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(
        model, source_loader, target_loader, optimizer, criterion_reg, criterion_domain,
        num_epochs=num_epochs, alpha=alpha, loss_threshold=loss_threshold, consecutive_epochs=consecutive_epochs
    )

    train_preds, train_targets, rmse_train, r2_train = evaluate_model(trained_model, source_loader, scaler_y)
    val_preds, val_targets, rmse_val, r2_val = evaluate_model(trained_model, target_loader, scaler_y)

    print(f"Train Set RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
    print(f"Val Set RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}")

    plot_predictions(train_preds, train_targets, val_preds, val_targets,
                     rmse_train, r2_train, rmse_val, r2_val, alpha)

    plot_tsne(trained_model, source_loader, target_loader)
    model_path = os.path.join(os.getcwd(), 'dann_model.pth')
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
