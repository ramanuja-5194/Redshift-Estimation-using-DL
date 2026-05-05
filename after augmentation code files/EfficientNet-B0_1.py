import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import random
import copy

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

DATA_DIR = "data/extracted_data/final"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess(data_dir):
    print("Loading data...")
    X_train = np.load(f"{data_dir}/X_train.npy").astype(np.float32)
    y_train = np.load(f"{data_dir}/y_train.npy").astype(np.float32)
    X_test = np.load(f"{data_dir}/X_test.npy").astype(np.float32)
    y_test = np.load(f"{data_dir}/y_test.npy").astype(np.float32)

    y_train = y_train[:, 0].reshape(-1, 1)
    y_test = y_test[:, 0].reshape(-1, 1)

    X_train = np.arcsinh(X_train)
    X_test = np.arcsinh(X_test)

    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train.std(axis=(0, 1, 2), keepdims=True)

    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_and_preprocess(DATA_DIR)

# ===== ROBUST STRATIFIED TRAIN / VAL SPLIT =====
percentiles = np.linspace(0, 100, 20)
bins = np.percentile(y_train, percentiles)
bins = np.unique(bins)
y_binned = np.digitize(y_train, bins)

unique_classes, class_counts = np.unique(y_binned, return_counts=True)
for cls, count in zip(unique_classes, class_counts):
    if count < 2:
        replacement = cls - 1 if cls > 1 else cls + 1
        y_binned[y_binned == cls] = replacement

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_binned
)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== DATA AUGMENTATION =====
def augment(x: torch.Tensor) -> torch.Tensor:
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k=k, dims=[2, 3])
    if random.random() > 0.5:
        x = torch.flip(x, dims=[3])
    if random.random() > 0.5:
        x = torch.flip(x, dims=[2])
    return x

class AstroEfficientNet(nn.Module):
    def __init__(self):
        super(AstroEfficientNet, self).__init__()
        self.effnet = models.efficientnet_b0(weights=None)
        
        old_conv = self.effnet.features[0][0]
        self.effnet.features[0][0] = nn.Conv2d(
            in_channels=5, out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
            padding=old_conv.padding, bias=False
        )
        
        num_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.effnet(x)

def calculate_astronomy_metrics(y_true, y_pred):
    dz_norm = (y_true - y_pred) / (1.0 + y_true)
    mean_bias = np.mean(dz_norm)
    median_bias = np.median(dz_norm)
    sigma_68 = 0.5 * (np.percentile(dz_norm, 84) - np.percentile(dz_norm, 16))
    sigma_std = np.std(dz_norm)
    outlier_rate_15 = np.mean(np.abs(dz_norm) > 0.15) * 100 
    
    return {
        "Mean Bias": mean_bias, "Median Bias": median_bias,
        "Sigma 68": sigma_68, "Sigma Std": sigma_std,
        "Outlier Rate (>0.15) %": outlier_rate_15
    }

model = AstroEfficientNet().to(DEVICE)
criterion = nn.HuberLoss(delta=0.1) 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

print("\nStarting Training...")
train_losses, val_losses = [], []
best_val_loss = float('inf')
best_model_weights = None

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        inputs = augment(inputs) 
        labels = labels.to(DEVICE)

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
    val_preds, val_actuals = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            val_preds.extend(outputs.cpu().numpy())
            val_actuals.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    v_preds_flat = np.array(val_preds).flatten()
    v_actuals_flat = np.array(val_actuals).flatten()
    astro_metrics = calculate_astronomy_metrics(v_actuals_flat, v_preds_flat)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(best_model_weights, 'efficientnet_best_weights.pkl')
        status = "(New Best Saved)"
    else:
        status = ""

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} {status}")

print("\nLoading Best Model Weights for Final Evaluation...")
model.load_state_dict(best_model_weights)

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
final_metrics = calculate_astronomy_metrics(actuals, predictions)

print(f"\n--- FINAL TEST RESULTS ---")
print(f"MSE: {mse:.5f}")
print(f"MAE: {mae:.5f}")
print(f"Sigma 68: {final_metrics['Sigma 68']:.5f}")
print(f"Outlier Rate (>0.15): {final_metrics['Outlier Rate (>0.15) %']:.2f}%")

# ==========================================
# 7. PLOTS
# ==========================================
print("\nSaving Plots...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss (Huber)')
plt.plot(val_losses, label='Val Loss (Huber)')
plt.xlabel('Epochs')
plt.ylabel('Huber Loss')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.scatter(actuals, predictions, s=1, alpha=0.1)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Spectroscopic z')
plt.ylabel('Predicted z')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title(f'Predicted vs True (Sigma_68={final_metrics["Sigma 68"]:.4f})')

plt.tight_layout()
plt.savefig("efficientnet_results.png")
print("Saved efficientnet_results.png")