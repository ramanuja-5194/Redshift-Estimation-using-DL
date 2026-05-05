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
import time
import math

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
DATA_DIR      = "data/extracted_data/final"
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4      # ViT needs a lower LR than CNNs
EPOCHS        = 50
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_MIN   = 0.05
Z_MAX   = 1.00
N_BINS  = 180

# --- ViT-specific hyperparameters ---
IMG_SIZE    = 40     # input image: 40x40 px
PATCH_SIZE  = 5      # each patch: 5x5 px  -> (40/5)^2 = 64 patches
N_PATCHES   = (IMG_SIZE // PATCH_SIZE) ** 2          # 64
PATCH_DIM   = 5 * PATCH_SIZE * PATCH_SIZE            # 5 bands * 25 px = 125
EMBED_DIM   = 256    # transformer embedding dimension
N_HEADS     = 8      # attention heads (EMBED_DIM must be divisible by N_HEADS)
N_LAYERS    = 6      # transformer encoder depth
MLP_RATIO   = 4      # hidden dim in FFN = EMBED_DIM * MLP_RATIO
DROPOUT     = 0.1

print(f"Using device: {DEVICE}")
print(f"Patch grid: {IMG_SIZE//PATCH_SIZE}x{IMG_SIZE//PATCH_SIZE} = {N_PATCHES} patches")
print(f"Patch token dim: {PATCH_DIM}  ->  Embed dim: {EMBED_DIM}")

BIN_EDGES   = np.linspace(Z_MIN, Z_MAX, N_BINS + 1)
BIN_CENTRES = torch.tensor(
    0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:]), dtype=torch.float32
).to(DEVICE)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def z_to_bin(z_array):
    indices = np.floor(
        (z_array - Z_MIN) / (Z_MAX - Z_MIN) * N_BINS
    ).astype(int)
    return np.clip(indices, 0, N_BINS - 1)

def augment(x):
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k=k, dims=[2, 3])
    if random.random() > 0.5:
        x = torch.flip(x, dims=[3])
    if random.random() > 0.5:
        x = torch.flip(x, dims=[2])
    return x

def calculate_astronomy_metrics(y_true, y_pred):
    dz = (y_true - y_pred) / (1.0 + y_true)
    return {
        "Mean Bias":   np.mean(dz),
        "Median Bias": np.median(dz),
        "Sigma 68":    0.5 * (np.percentile(dz, 84) - np.percentile(dz, 16)),
        "Sigma Std":   np.std(dz),
        "Outlier %":   np.mean(np.abs(dz) > 0.15) * 100,
        "sigma_MAD":   1.4826 * np.median(np.abs(dz - np.median(dz))),
    }

# ==========================================
# 3. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess(data_dir):
    print("Loading data...")
    X_train = np.load(f"{data_dir}/X_train.npy").astype(np.float32)
    y_train = np.load(f"{data_dir}/y_train.npy").astype(np.float32)
    X_test  = np.load(f"{data_dir}/X_test.npy").astype(np.float32)
    y_test  = np.load(f"{data_dir}/y_test.npy").astype(np.float32)

    z_train   = y_train[:, 0].reshape(-1, 1)
    z_test    = y_test[:, 0].reshape(-1, 1)
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

    print(f"Data ready. X: {X_train.shape}")
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

def make_balanced_sampler(bin_labels):
    flat   = bin_labels.flatten()
    counts = np.bincount(flat, minlength=N_BINS).astype(float)
    counts = np.maximum(counts, 1.0)
    w      = (1.0 / counts)[flat]
    return WeightedRandomSampler(torch.from_numpy(w).float(),
                                  num_samples=len(w), replacement=True)

train_sampler = make_balanced_sampler(bin_train)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(bin_train.flatten(), dtype=torch.long), torch.tensor(z_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(bin_val.flatten(),   dtype=torch.long), torch.tensor(z_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(bin_test.flatten(),  dtype=torch.long), torch.tensor(z_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 4. VISION TRANSFORMER ARCHITECTURE
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=5, embed_dim=EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2

        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        patch_in_dim = 64 * patch_size * patch_size
        self.projection = nn.Linear(patch_in_dim, embed_dim)

    def forward(self, x):
        x = self.cnn_stem(x)                         
        B, C, H, W = x.shape
        P = self.patch_size

        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)             
        x = x.reshape(B, (H // P) * (W // P), -1)   

        x = self.projection(x)                       
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale    = self.head_dim ** -0.5   

        self.qkv     = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)     
        q, k, v = qkv.unbind(0)               

        attn  = (q @ k.transpose(-2, -1)) * self.scale   
        attn  = F.softmax(attn, dim=-1)
        attn  = self.dropout(attn)

        out = (attn @ v).transpose(1, 2)      
        out = out.reshape(B, N, D)            
        return self.proj(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=N_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, n_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   
        x = x + self.ffn(self.norm2(x))    
        return x

class AstroViT(nn.Module):
    def __init__(self, n_bins=N_BINS):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=5, embed_dim=EMBED_DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, N_PATCHES + 1, EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        self.pos_dropout = nn.Dropout(DROPOUT)

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(EMBED_DIM, N_HEADS, MLP_RATIO, DROPOUT)
            for _ in range(N_LAYERS)
        ])

        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_bins)
        )

    def forward(self, x):
        B = x.size(0)
        tokens = self.patch_embed(x)                     
        cls = self.cls_token.expand(B, -1, -1)           
        tokens = torch.cat([cls, tokens], dim=1)         
        tokens = self.pos_dropout(tokens + self.pos_embed)
        tokens = self.encoder(tokens)                    
        cls_out = self.norm(tokens[:, 0])                
        logits  = self.head(cls_out)                     
        return logits

    def predict_z(self, logits):
        probs  = F.softmax(logits, dim=1)
        z_pred = (probs * BIN_CENTRES.unsqueeze(0)).sum(1)
        return z_pred

    def predict_pdf(self, logits):
        return F.softmax(logits, dim=1)

    def get_attention_maps(self, x):
        B = x.size(0)
        tokens = self.patch_embed(x)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        first_block = self.encoder[0]
        normed   = first_block.norm1(tokens)
        qkv      = first_block.attn.qkv(normed)
        B2, N, _ = qkv.shape
        qkv      = qkv.reshape(B2, N, 3, N_HEADS, EMBED_DIM // N_HEADS)
        qkv      = qkv.permute(2, 0, 3, 1, 4)
        q, k, _  = qkv.unbind(0)
        scale    = (EMBED_DIM // N_HEADS) ** -0.5
        attn     = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        return attn[:, :, 0, 1:].mean(1)  

# ==========================================
# 5. INITIALISE MODEL
# ==========================================
print("\nBuilding AstroViT...")
model     = AstroViT(n_bins=N_BINS).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

def lr_lambda(epoch):
    warmup = 5
    if epoch < warmup:
        return epoch / warmup
    cos_progress = (epoch - warmup) / (EPOCHS - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * cos_progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

# ==========================================
# 6. TRAINING LOOP
# ==========================================
print("\nStarting Training...")
train_losses, val_losses = [], []
best_val_loss      = float('inf')
best_model_weights = None
epoch_times        = []

for epoch in range(EPOCHS):
    t_epoch_start = time.perf_counter()

    model.train()
    running_loss = 0.0

    for imgs, bin_labels, _ in train_loader:
        imgs       = augment(imgs.to(DEVICE))
        bin_labels = bin_labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, bin_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    scheduler.step()

    model.eval()
    val_loss    = 0.0
    val_preds   = []
    val_actuals = []

    with torch.no_grad():
        for imgs, bin_labels, z_true in val_loader:
            imgs       = imgs.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE)
            logits     = model(imgs)
            val_loss  += criterion(logits, bin_labels).item() * imgs.size(0)
            z_pred     = model.predict_z(logits)
            val_preds.extend(z_pred.cpu().numpy())
            val_actuals.extend(z_true.numpy().flatten())

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    metrics = calculate_astronomy_metrics(
        np.array(val_actuals), np.array(val_preds)
    )

    t_epoch_end = time.perf_counter()
    epoch_times.append(t_epoch_end - t_epoch_start)
    remaining   = EPOCHS - (epoch + 1)
    eta_sec     = np.mean(epoch_times) * remaining
    eta_h, r    = divmod(int(eta_sec), 3600)
    eta_m, eta_s = divmod(r, 60)

    current_lr = optimizer.param_groups[0]['lr']

    if val_loss < best_val_loss:
        best_val_loss      = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(best_model_weights, 'vit_best_weights.pkl')
        status = "✓ Best Saved"
    else:
        status = ""

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
        f"σ68: {metrics['Sigma 68']:.4f} | "
        f"Outliers: {metrics['Outlier %']:.2f}% | "
        f"LR: {current_lr:.2e} | "
        f"ETA: {eta_h}h{eta_m:02d}m  {status}"
    )

print("\nLoading best model weights...")
model.load_state_dict(best_model_weights)

total_train_time = sum(epoch_times)
print(f"\nActual total training time: {total_train_time/3600:.2f} h  "
      f"({total_train_time/60:.1f} min)")


# ==========================================
# 7. TEST EVALUATION
# ==========================================
print("\nEvaluating on Test Set...")
model.eval()
all_preds, all_actuals, all_pdfs = [], [], []

with torch.no_grad():
    for imgs, _, z_true in test_loader:
        logits = model(imgs.to(DEVICE))
        z_pred = model.predict_z(logits)
        pdf    = model.predict_pdf(logits)
        all_preds.extend(z_pred.cpu().numpy())
        all_actuals.extend(z_true.numpy().flatten())
        all_pdfs.append(pdf.cpu().numpy())

predictions = np.array(all_preds)
actuals     = np.array(all_actuals)
all_pdfs    = np.vstack(all_pdfs)

final_metrics = calculate_astronomy_metrics(actuals, predictions)

print("\n" + "="*55)
print("  FINAL TEST RESULTS — AstroViT")
print("="*55)
for k, v in final_metrics.items():
    print(f"  {k:<22}: {v:+.5f}")
print("="*55)


# ==========================================
# 8. PLOTS
# ==========================================
print("\nSaving Plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("AstroViT — Photometric Redshift Results", fontsize=14)

axes[0].plot(train_losses, label='Train (CE)')
axes[0].plot(val_losses,   label='Val (CE)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Training History'); axes[0].legend()

axes[1].scatter(actuals, predictions, s=1, alpha=0.1, c='teal')
axes[1].plot([Z_MIN, Z_MAX], [Z_MIN, Z_MAX], 'r--', lw=1.5)
axes[1].set_xlabel('Spec-z (true)'); axes[1].set_ylabel('Predicted z')
axes[1].set_title(
    f'σ68={final_metrics["Sigma 68"]:.4f}  '
    f'Outliers={final_metrics["Outlier %"]:.2f}%'
)
axes[1].set_xlim(Z_MIN, Z_MAX); axes[1].set_ylim(Z_MIN, Z_MAX)

dz_norm = (actuals - predictions) / (1.0 + actuals)
axes[2].hist(dz_norm, bins=100, color='teal', alpha=0.8, density=True)
axes[2].axvline(0,     color='red',    ls='--', lw=1.5, label='Zero bias')
axes[2].axvline( 0.15, color='orange', ls=':',  lw=1.5, label='±0.15 outlier')
axes[2].axvline(-0.15, color='orange', ls=':',  lw=1.5)
axes[2].set_xlabel(r'$\Delta z/(1+z_{\rm true})$')
axes[2].set_ylabel('Density'); axes[2].set_xlim(-0.5, 0.5)
axes[2].set_title('Normalised Residuals'); axes[2].legend()

plt.tight_layout()
plt.savefig("vit_results.png", dpi=150)
print("Saved vit_results.png")


# ==========================================
# 9. BONUS: ATTENTION MAP VISUALISATION
# ==========================================
print("\nGenerating attention map visualisations...")

model.eval()
sample_imgs, _, sample_z = next(iter(test_loader))
sample_imgs = sample_imgs[:8].to(DEVICE)

with torch.no_grad():
    attn_maps = model.get_attention_maps(sample_imgs)  # (8, N_patches)
    logits    = model(sample_imgs)
    z_preds   = model.predict_z(logits).cpu().numpy()

P    = IMG_SIZE // PATCH_SIZE   
attn = attn_maps.cpu().numpy().reshape(-1, P, P)

fig, axes = plt.subplots(3, 8, figsize=(20, 8))
fig.suptitle("AstroViT Attention Maps — What the model attends to", fontsize=13)

z_true_sample = sample_z[:8].numpy().flatten()

for i in range(8):
    # Row 0: r-band galaxy image
    axes[0, i].imshow(sample_imgs[i, 2].cpu().numpy(), cmap='magma', origin='lower')
    axes[0, i].set_title(f"z_true={z_true_sample[i]:.3f}", fontsize=8)
    axes[0, i].axis('off')

    # Row 1: attention heatmap
    axes[1, i].imshow(attn[i], cmap='hot', origin='lower',
                      interpolation='bilinear')
    axes[1, i].set_title(f"z_pred={z_preds[i]:.3f}", fontsize=8)
    axes[1, i].axis('off')

    # Row 2: overlay (Using PyTorch native interpolation instead of cv2)
    r_band = sample_imgs[i, 2].cpu().numpy()
    r_norm = (r_band - r_band.min()) / (r_band.max() - r_band.min() + 1e-9)
    
    attn_tensor = torch.tensor(attn[i]).unsqueeze(0).unsqueeze(0) 
    attn_resized = F.interpolate(attn_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze().numpy()
    
    axes[2, i].imshow(r_norm, cmap='gray', origin='lower')
    axes[2, i].imshow(attn_resized, cmap='hot', alpha=0.5,
                      origin='lower', interpolation='bilinear')
    axes[2, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel("r-band", fontsize=9)
        axes[1, i].set_ylabel("Attention", fontsize=9)
        axes[2, i].set_ylabel("Overlay", fontsize=9)

plt.tight_layout()
plt.savefig("vit_attention_maps.png", dpi=150)
print("Saved vit_attention_maps.png")