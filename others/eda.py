import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("Loading data...")
X = np.load('data/extracted_data/final/X_test.npy')
y = np.load('data/extracted_data/final/y_test.npy')

print(f"X Data Shape: {X.shape} (Count, Height, Width, Bands)")
print(f"y Data Shape: {y.shape} (Count, Features)")

# Extract columns for clarity
# y columns: [z, zErr, photo_z, photo_zErr]
z_spec = y[:, 0]        # Ground Truth Redshift
z_err = y[:, 1]         # Error in Ground Truth
z_photo_sdss = y[:, 2]  # SDSS Baseline Prediction

# Plot 1: Distribution of Redshift (z)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(z_spec, bins=50, kde=True, color='blue')
plt.title('Distribution of Spectroscopic Redshift (Target)')
plt.xlabel('Redshift (z)')
plt.show()

# Plot 2: Accuracy of existing SDSS Photo-z (Baseline)
plt.subplot(1, 3, 2)
plt.scatter(z_spec, z_photo_sdss, alpha=0.1, s=1, c='green')
plt.plot([0, 1], [0, 1], 'r--', lw=2)  # Perfect prediction line
plt.title('Baseline: Spec-z vs SDSS Photo-z')
plt.xlabel('True z')
plt.ylabel('SDSS Predicted z')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# Plot 3: Distribution of Errors
plt.subplot(1, 3, 3)
sns.histplot(z_err, bins=50, color='red', log_scale=(False, True))
plt.title('Distribution of Spectroscopic Error (log scale)')
plt.xlabel('Error Value')

plt.tight_layout()
plt.show()

def plot_galaxy(index):
    """
    Plots the 5 individual bands and a composite 'RGB' image for a specific galaxy.
    The 'RGB' is simulated using i, r, g bands.
    """
    img = X[index]
    target_z = z_spec[index]

    # Create a rough RGB composite using (i, r, g) mapped to (R, G, B)
    # We use arcsinh scaling to make faint structures visible
    rgb = np.zeros((40, 40, 3))
    rgb[:,:,0] = np.arcsinh(img[:,:,3]) # Red channel = i band
    rgb[:,:,1] = np.arcsinh(img[:,:,2]) # Green channel = r band
    rgb[:,:,2] = np.arcsinh(img[:,:,1]) # Blue channel = g band

    # Normalize RGB to 0-1 for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9)

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    bands = ['u (UV)', 'g (Green)', 'r (Red)', 'i (Near IR)', 'z (IR)']

    # Plot Composite
    axes[0].imshow(rgb)
    axes[0].set_title(f"ID: {index}\nTrue z: {target_z:.3f}")
    axes[0].axis('off')

    # Plot Individual Bands
    for i in range(5):
        # We use a 'magma' colormap which is good for intensity
        axes[i+1].imshow(img[:,:,i], cmap='magma', origin='lower')
        axes[i+1].set_title(bands[i])
        axes[i+1].axis('off')

    plt.show()

# Visualize 3 random galaxies
import random
random_indices = random.sample(range(len(X)), 3)
for idx in random_indices:
    plot_galaxy(idx)


# 4. BASIC STATS & OUTLIER CHECK
print("\n--- Statistics ---")
print(f"Max pixel value in dataset: {np.max(X):.2f}")
print(f"Min pixel value in dataset: {np.min(X):.2f}")
print(f"Mean pixel value: {np.mean(X):.2f}")
print(f"Redshift Range: {np.min(z_spec):.4f} to {np.max(z_spec):.4f}")

# Check for 'dead' pixels or empty images
empty_images = np.where(np.sum(X, axis=(1,2,3)) == 0)[0]
print(f"Number of completely empty images: {len(empty_images)}")

print(f"First 2 elements: {X[:2]}")