import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import random

# 0. REPRODUCIBILITY 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# 1. CONFIGURATION
DATA_DIR = "data/extracted_data/final"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# 2. DATA LOADING & PREPROCESSING
def load_and_preprocess(data_dir):
    print("Loading data...")
    X_train = np.load(f"{data_dir}/X_train.npy").astype(np.float32)
    y_train = np.load(f"{data_dir}/y_train.npy").astype(np.float32)
    X_test = np.load(f"{data_dir}/X_test.npy").astype(np.float32)
    y_test = np.load(f"{data_dir}/y_test.npy").astype(np.float32)

    # Target: spectroscopic redshift z
    y_train = y_train[:, 0].reshape(-1, 1)
    y_test = y_test[:, 0].reshape(-1, 1)

    # Arcsinh scaling
    X_train = np.arcsinh(X_train)
    X_test = np.arcsinh(X_test)

    # ===== PER-CHANNEL NORMALIZATION =====
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train.std(axis=(0, 1, 2), keepdims=True)

    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    # Channel swap (NHWC → NCHW)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    print(f"Data Prepared. X Shape: {X_train.shape}, y Shape: {y_train.shape}")
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_and_preprocess(DATA_DIR)

# ===== TRAIN / VAL SPLIT =====
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. DEFINE MODEL
class GalaxyCNN(nn.Module):
    def __init__(self):
        super(GalaxyCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)

model = GalaxyCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== LR SCHEDULER =====
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

# 4. TRAINING LOOP
print("\nStarting Training...")
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# 5. EVALUATION (TEST SET ONLY)
print("\nEvaluating on Test Set...")
model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(labels.numpy())

predictions = np.array(predictions).flatten()
actuals = np.array(actuals).flatten()

mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

print(f"\nFinal Test MSE: {mse:.5f}")
print(f"Final Test MAE: {mae:.5f}")

# 6. PLOTS
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.scatter(actuals, predictions, s=1, alpha=0.1)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Spectroscopic z')
plt.ylabel('Predicted z')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title(f'Predicted vs True (MAE={mae:.4f})')

plt.tight_layout()
plt.savefig("my_model_results.png")
plt.show()
