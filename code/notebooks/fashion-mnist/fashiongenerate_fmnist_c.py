import numpy as np
from datasets import load_dataset
import os
import mnist_cc as mnist_c

# =========================
# 1. Load Fashion-MNIST
# =========================
ds = load_dataset("fashion_mnist")

X = np.array([np.array(x['image']) for x in ds['test']])
y = np.array([x['label'] for x in ds['test']])

print("Loaded dataset:", X.shape)   # (10000, 28, 28)

# =========================
# 2. Create output folder
# =========================
save_path = "fashion-mnist-c"
os.makedirs(save_path, exist_ok=True)

# =========================
# 3. SVHN-C ALIGNED CORRUPTIONS
# =========================
corruptions = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",

    "gaussian_blur",
    "motion_blur",
    "zoom_blur",

    "fog",
    "snow",
    "spatter",

    "brightness",
    "contrast",
    "jpeg_compression",

    "translate",
    "rotate",
    "scale"
]

# =========================
# 4. Generate corrupted data
# =========================
for name in corruptions:

    print(f"\nProcessing corruption: {name}")

    if not hasattr(mnist_c, name):
        print(f"❌ Missing function: {name}")
        continue

    func = getattr(mnist_c, name)
    all_severity_data = []

    for severity in range(1, 6):

        corrupted_images = []

        for img in X:
            c_img = func(img, severity)
            corrupted_images.append(c_img)

        corrupted_images = np.array(corrupted_images, dtype=np.float32)
        all_severity_data.append(corrupted_images)

        print(f"  Severity {severity} done")

    final_data = np.concatenate(all_severity_data, axis=0)

    np.save(os.path.join(save_path, f"{name}.npy"), final_data)

    print(f"✅ Saved: {name}.npy")

# =========================
# 5. Save labels
# =========================
labels = np.tile(y, 5)
np.save(os.path.join(save_path, "labels.npy"), labels)

print("\n✅ DONE: Fashion-MNIST-C aligned with SVHN-C!")