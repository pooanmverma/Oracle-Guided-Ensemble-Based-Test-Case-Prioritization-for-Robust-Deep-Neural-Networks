import os
import numpy as np
from torchvision.datasets import SVHN
from torchvision import transforms

# import your corruption file
from corruptions_svhn_final import CORRUPTIONS

# -------------------------
# Config
# -------------------------
SAVE_DIR = "./svhn_c"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------
# Load dataset
# -------------------------
dataset = SVHN(
    root="./data",
    split="test",
    download=True,
    transform=None
)

NUM_IMAGES = len(dataset)
# -------------------------
# Save labels
# -------------------------
labels = np.array([dataset[i][1] for i in range(NUM_IMAGES)])
np.save(f"{SAVE_DIR}/labels.npy", labels)

# -------------------------
# Generate corruptions
# -------------------------
for corruption in CORRUPTIONS:
    name = corruption.__name__
    print(f"\nProcessing: {name}")

    for severity in range(1, 6):
        print(f"  Severity {severity}")

        corrupted_images = []

        for i in range(NUM_IMAGES):
            img, _ = dataset[i]

            corrupted = corruption(img, severity=severity)
            corrupted_images.append(corrupted)

        corrupted_images = np.array(corrupted_images)

        np.save(f"{SAVE_DIR}/{name}_{severity}.npy", corrupted_images)

print("\n✅ SVHN-C generation complete!")