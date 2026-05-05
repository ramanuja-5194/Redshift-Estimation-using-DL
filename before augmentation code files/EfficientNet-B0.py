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

# ==========================================
# 0. REPRODUCIBILITY 
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = "data/extracted_data/final"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
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

# ==========================================
# 3. DEFINE MODEL: ASTRO-EFFICIENTNET-B0
# ==========================================
class AstroEfficientNet(nn.Module):
    def __init__(self):
        super(AstroEfficientNet, self).__init__()
        
        # Load standard EfficientNet-B0 (Untrained)
        self.effnet = models.efficientnet_b0(weights=None)
        
        # SURGERY STEP 1: Modify input layer for 5 Channels
        # EfficientNet's first conv layer is inside features[0][0]
        old_conv = self.effnet.features[0][0]
        self.effnet.features[0][0] = nn.Conv2d(
            in_channels=5, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=False
        )
        
        # SURGERY STEP 2: Modify output layer for Regression
        # EfficientNet classifier has Dropout at index 0 and Linear at index 1
        num_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(num_features, 1) # Output single Redshift value

    def forward(self, x):
        return self.effnet(x)

# ==========================================
# 4. CUSTOM LOSS AND METRICS
# ==========================================
def calculate_astronomy_metrics(y_true, y_pred):
    """
    Calculates non-differentiable evaluation metrics to show at the end of epochs.
    Expects 1D numpy arrays.
    """
    # Normalized error delta z
    dz_norm = (y_true - y_pred) / (1.0 + y_true)
    
    # Biases
    mean_bias = np.mean(dz_norm)
    median_bias = np.median(dz_norm)
    
    # Robust Scatter (Sigma 68)
    sigma_68 = 0.5 * (np.percentile(dz_norm, 84) - np.percentile(dz_norm, 16))
    
    # Standard Deviation
    sigma_std = np.std(dz_norm)
    
    # Outlier Fractions (eta)
    outlier_rate_15 = np.mean(np.abs(dz_norm) > 0.15) * 100 
    
    return {
        "Mean Bias": mean_bias,
        "Median Bias": median_bias,
        "Sigma 68": sigma_68,
        "Sigma Std": sigma_std,
        "Outlier Rate (>0.15) %": outlier_rate_15
    }

# Initialize Model, Huber Loss, and Optimizer
model = AstroEfficientNet().to(DEVICE)

# Huber Loss is excellent for astronomy datasets to prevent outliers from ruining gradients
criterion = nn.HuberLoss(delta=0.1) 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== LR SCHEDULER =====
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

# ==========================================
# 5. TRAINING LOOP WITH EARLY STOPPING
# ==========================================
print("\nStarting Training...")
train_losses, val_losses = [], []

best_val_loss = float('inf')
best_model_weights = None

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
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

    # --- VALIDATION PHASE ---
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

    # Calculate Astro Metrics for Validation
    v_preds_flat = np.array(val_preds).flatten()
    v_actuals_flat = np.array(val_actuals).flatten()
    astro_metrics = calculate_astronomy_metrics(v_actuals_flat, v_preds_flat)

    # ===== EARLY STOPPING CHECK =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        status = "(New Best Saved)"
    else:
        status = ""

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} {status}")
    print(f"       -> Sig68: {astro_metrics['Sigma 68']:.4f} | Outliers: {astro_metrics['Outlier Rate (>0.15) %']:.2f}% | Mean Bias: {astro_metrics['Mean Bias']:.4f}")

# Load the best weights before final testing
print("\nLoading Best Model Weights for Final Evaluation...")
model.load_state_dict(best_model_weights)

# ==========================================
# 6. EVALUATION (TEST SET ONLY)
# ==========================================
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

# Standard ML Metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

# Astronomy Specific Metrics
final_metrics = calculate_astronomy_metrics(actuals, predictions)

print(f"\n--- FINAL TEST RESULTS ---")
print(f"MSE: {mse:.5f}")
print(f"MAE: {mae:.5f}")
print(f"Sigma 68: {final_metrics['Sigma 68']:.5f}")
print(f"Outlier Rate (>0.15): {final_metrics['Outlier Rate (>0.15) %']:.2f}%")
print(f"Mean Bias: {final_metrics['Mean Bias']:.5f}")
print(f"Median Bias: {final_metrics['Median Bias']:.5f}")

# ==========================================
# 7. PLOTS
# ==========================================
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
plt.savefig("efficientnet_model_results.png")
plt.show()