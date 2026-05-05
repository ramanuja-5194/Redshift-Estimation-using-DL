import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import copy

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

DATA_DIR     = "data/extracted_data/final"
BATCH_SIZE   = 64
LEARNING_RATE = 5e-4
EPOCHS       = 50
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_MIN        = 0.05    
Z_MAX        = 1.00    
N_BINS       = 180     

BIN_EDGES    = np.linspace(Z_MIN, Z_MAX, N_BINS + 1)
BIN_CENTRES  = torch.tensor(
    0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:]), dtype=torch.float32
).to(DEVICE)

def z_to_bin(z_array: np.ndarray) -> np.ndarray:
    indices = np.floor((z_array - Z_MIN) / (Z_MAX - Z_MIN) * N_BINS).astype(int)
    return np.clip(indices, 0, N_BINS - 1)

def load_and_preprocess(data_dir):
    print("Loading data...")
    X_train = np.load(f"{data_dir}/X_train.npy").astype(np.float32)
    y_train = np.load(f"{data_dir}/y_train.npy").astype(np.float32)
    X_test  = np.load(f"{data_dir}/X_test.npy").astype(np.float32)
    y_test  = np.load(f"{data_dir}/y_test.npy").astype(np.float32)

    z_train = y_train[:, 0].reshape(-1, 1)
    z_test  = y_test[:, 0].reshape(-1, 1)

    bin_train = z_to_bin(y_train[:, 0]).reshape(-1, 1)
    bin_test  = z_to_bin(y_test[:, 0]).reshape(-1, 1)

    X_train = np.arcsinh(X_train)
    X_test  = np.arcsinh(X_test)

    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std  = X_train.std(axis=(0, 1, 2),  keepdims=True)

    X_train = (X_train - mean) / (std + 1e-7)
    X_test  = (X_test  - mean) / (std + 1e-7)

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test  = np.transpose(X_test,  (0, 3, 1, 2))

    return X_train, z_train, bin_train, X_test, z_test, bin_test

X_train, z_train, bin_train, X_test, z_test, bin_test = load_and_preprocess(DATA_DIR)

# ===== ROBUST STRATIFIED TRAIN / VAL SPLIT (20%) =====
stratify_labels = bin_train.copy()
unique_classes, class_counts = np.unique(stratify_labels, return_counts=True)
for cls, count in zip(unique_classes, class_counts):
    if count < 2:
        replacement = cls - 1 if cls > 0 else cls + 1
        stratify_labels[stratify_labels == cls] = replacement

(X_train, X_val,
 z_train, z_val,
 bin_train, bin_val) = train_test_split(
    X_train, z_train, bin_train,
    test_size=0.2, random_state=42, stratify=stratify_labels
)

def make_balanced_sampler(bin_labels: np.ndarray) -> WeightedRandomSampler:
    bin_labels_flat = bin_labels.flatten()
    counts = np.bincount(bin_labels_flat, minlength=N_BINS).astype(float)
    counts = np.maximum(counts, 1.0)                    
    weight_per_bin = 1.0 / counts                       
    sample_weights = weight_per_bin[bin_labels_flat]    
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )

train_sampler = make_balanced_sampler(bin_train)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(bin_train.flatten(), dtype=torch.long), torch.tensor(z_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(bin_val.flatten(), dtype=torch.long), torch.tensor(z_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(bin_test.flatten(), dtype=torch.long), torch.tensor(z_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_1x1: int, out_3x3: int, out_5x5: int, out_pool: int):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_ch, out_1x1, kernel_size=1), nn.BatchNorm2d(out_1x1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_ch, out_3x3 // 2, kernel_size=1), nn.BatchNorm2d(out_3x3 // 2), nn.ReLU(inplace=True), nn.Conv2d(out_3x3 // 2, out_3x3, kernel_size=3, padding=1), nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_ch, out_5x5 // 2, kernel_size=1), nn.BatchNorm2d(out_5x5 // 2), nn.ReLU(inplace=True), nn.Conv2d(out_5x5 // 2, out_5x5, kernel_size=5, padding=2), nn.BatchNorm2d(out_5x5), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(in_ch, out_pool, kernel_size=1), nn.BatchNorm2d(out_pool), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class PasquetPhotoZ(nn.Module):
    def __init__(self, n_bins: int = N_BINS):
        super().__init__()
        self.n_bins = n_bins
        self.stem = nn.Sequential(nn.Conv2d(5, 64, kernel_size=5, padding=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.inception1 = InceptionBlock(in_ch=64, out_1x1=32, out_3x3=48, out_5x5=24, out_pool=24)
        self.inception2 = InceptionBlock(in_ch=128, out_1x1=64, out_3x3=96, out_5x5=48, out_pool=48)
        self.pool_mid = nn.MaxPool2d(2, 2)
        self.inception3 = InceptionBlock(in_ch=256, out_1x1=96, out_3x3=144, out_5x5=72, out_pool=72)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(384, 512), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(512, n_bins))

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool_mid(x)
        x = self.inception3(x)
        x = self.gap(x)
        return self.classifier(x)

    def predict_z(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)                    
        return (probs * BIN_CENTRES.unsqueeze(0)).sum(1)
        
    def predict_pdf(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=1)

def calculate_astronomy_metrics(y_true, y_pred):
    dz = (y_true - y_pred) / (1.0 + y_true)
    return {
        "Mean Bias": np.mean(dz), "Median Bias": np.median(dz),
        "Sigma 68": 0.5 * (np.percentile(dz, 84) - np.percentile(dz, 16)),
        "Sigma Std": np.std(dz), "Outlier %": np.mean(np.abs(dz) > 0.15) * 100,
        "sigma_MAD": 1.4826 * np.median(np.abs(dz - np.median(dz))),
    }

model = PasquetPhotoZ(n_bins=N_BINS).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print("\nStarting Training...")
train_losses, val_losses = [], []
best_val_loss = float('inf')
best_model_weights = None

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, bin_labels, _ in train_loader:
        imgs = augment(imgs.to(DEVICE))
        bin_labels = bin_labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, bin_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    val_preds, val_actuals = [], []

    with torch.no_grad():
        for imgs, bin_labels, z_true in val_loader:
            imgs, bin_labels = imgs.to(DEVICE), bin_labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, bin_labels)
            val_loss += loss.item() * imgs.size(0)

            z_pred = model.predict_z(logits)
            val_preds.extend(z_pred.cpu().numpy())
            val_actuals.extend(z_true.numpy().flatten())

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step()

    metrics = calculate_astronomy_metrics(np.array(val_actuals), np.array(val_preds))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(best_model_weights, 'pasquet_inception_best_weights.pkl')
        status = "✓ Best Saved"
    else:
        status = ""

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | σ68: {metrics['Sigma 68']:.4f} {status}")

print("\nLoading best model weights...")
model.load_state_dict(best_model_weights)

print("\nEvaluating on Test Set...")
model.eval()
all_preds, all_actuals, all_pdfs = [], [], []

with torch.no_grad():
    for imgs, _, z_true in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        z_pred = model.predict_z(logits)
        pdf    = model.predict_pdf(logits)

        all_preds.extend(z_pred.cpu().numpy())
        all_actuals.extend(z_true.numpy().flatten())
        all_pdfs.append(pdf.cpu().numpy())

predictions = np.array(all_preds)
actuals = np.array(all_actuals)
all_pdfs = np.vstack(all_pdfs)
final_metrics = calculate_astronomy_metrics(actuals, predictions)

print("\n  FINAL TEST RESULTS")
print(f"  Sigma 68:          {final_metrics['Sigma 68']:.5f}")
print(f"  Outlier Rate >0.15:{final_metrics['Outlier %']:.2f}%")


# ==========================================
# 8. PLOTS
# ==========================================
print("\nSaving Plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Pasquet-Style Inception Network — Photo-z Results", fontsize=14)

# Plot 1: Training history
ax = axes[0]
ax.plot(train_losses, label='Train Loss (CE)')
ax.plot(val_losses,   label='Val Loss (CE)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Training History')
ax.legend()

# Plot 2: Predicted vs true z
ax = axes[1]
ax.scatter(actuals, predictions, s=1, alpha=0.1, c='steelblue')
ax.plot([Z_MIN, Z_MAX], [Z_MIN, Z_MAX], 'r--', lw=1.5, label='Perfect')
ax.set_xlabel('Spectroscopic z (true)')
ax.set_ylabel('Predicted z (PDF mean)')
ax.set_xlim(Z_MIN, Z_MAX)
ax.set_ylim(Z_MIN, Z_MAX)
ax.set_title(
    f'σ68={final_metrics["Sigma 68"]:.4f}  '
    f'Outliers={final_metrics["Outlier %"]:.2f}%'
)
ax.legend()

# Plot 3: Normalised residual distribution
ax = axes[2]
dz_norm = (actuals - predictions) / (1.0 + actuals)
ax.hist(dz_norm, bins=100, color='steelblue', alpha=0.8, density=True)
ax.axvline(0,                    color='red', ls='--', lw=1.5, label='Zero bias')
ax.axvline( 0.15,                color='orange', ls=':', lw=1.5, label='±0.15 (outlier)')
ax.axvline(-0.15,                color='orange', ls=':', lw=1.5)
ax.set_xlabel(r'$\Delta z\,/\,(1+z_{\rm true})$')
ax.set_ylabel('Density')
ax.set_title('Normalised Residuals')
ax.set_xlim(-0.5, 0.5)
ax.legend()

plt.tight_layout()
plt.savefig("pasquet_inception_results.png", dpi=150)
print("Saved pasquet_inception_results.png")

# Bonus: Example PDFs
print("\nGenerating example PDF plots...")
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.suptitle("Example Redshift PDFs (Pasquet Network)", fontsize=13)

indices = np.random.choice(len(actuals), 8, replace=False)

for ax, idx in zip(axes.flatten(), indices):
    pdf  = all_pdfs[idx]
    z_t  = actuals[idx]
    z_p  = predictions[idx]
    ax.bar(BIN_CENTRES.cpu().numpy(), pdf,
           width=(Z_MAX - Z_MIN) / N_BINS,
           color='steelblue', alpha=0.7)
    ax.axvline(z_t, color='red',    lw=2, label=f'True z={z_t:.3f}')
    ax.axvline(z_p, color='orange', lw=2, ls='--', label=f'Pred z={z_p:.3f}')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('P(z)')
    ax.legend(fontsize=7)
    ax.set_xlim(Z_MIN, Z_MAX)

plt.tight_layout()
plt.savefig("pasquet_example_pdfs.png", dpi=150)
print("Saved pasquet_example_pdfs.png")