import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
OUTPUT_ROOT = "MOBIUS_LATEST_FID_GENERATED/Images"   # where output will be written
CHECKPOINT_PATH = "savedModels/cDCGAN_MOBIUS/latest_checkpoint.pth"

NUM_CLASSES = 100             # classes 1..100 inclusive
GEN_PER_CLASS = 50            # generate 50 images per class
Z_DIM = 100
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------
# LOAD GENERATOR (EDIT IMPORT BELOW)
# ------------------------------------------------------
class Generator(nn.Module):
    def __init__(
        self, z_dim=100, num_classes=100, embed_dim=100, img_channels=3, feature_g=64
    ):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.input_dim = z_dim + embed_dim
        self.init_size = 8  # starting feature map size
        self.fc = nn.Linear(
            self.input_dim, feature_g * 16 * self.init_size * self.init_size
        )

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )

        self.net = nn.Sequential(
            block(feature_g * 16, feature_g * 8),  # 8 -> 16
            block(feature_g * 8, feature_g * 4),  # 16 -> 32
            block(feature_g * 4, feature_g * 2),  # 32 -> 64
            block(feature_g * 2, feature_g),  # 64 -> 128
            block(feature_g, feature_g // 2),  # 128 -> 256
            nn.Conv2d(feature_g // 2, img_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # Get class embedding and concat with z
        emb = self.embed(labels)
        x = torch.cat([z, emb], dim=1)
        out = self.fc(x).view(x.size(0), -1, self.init_size, self.init_size)
        return self.net(out)


G = Generator(
    z_dim=Z_DIM,
    num_classes=NUM_CLASSES,
    img_channels=3,
    feature_g=64
).to(DEVICE)

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
G.load_state_dict(ckpt["generator_state_dict"])
G.eval()

print("✅ Loaded generator checkpoint.")


# ------------------------------------------------------
# NORMALIZATION / INV-NORMALIZATION
# ------------------------------------------------------
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.5]*3),
    transforms.Normalize(mean=[-0.5]*3, std=[1]*3),
])


# ------------------------------------------------------
# COUNT IMAGES IN A CLASS DIRECTORY
# ------------------------------------------------------
def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


# ------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------
os.makedirs(OUTPUT_ROOT, exist_ok=True)

for class_id in tqdm(range(1, NUM_CLASSES + 1), desc="Generating classes"):

    out_dir = os.path.join(OUTPUT_ROOT, str(class_id))
    os.makedirs(out_dir, exist_ok=True)

    before = count_images(out_dir)

    labels = torch.full((GEN_PER_CLASS,), class_id - 1, dtype=torch.long, device=DEVICE)
    z = torch.randn(GEN_PER_CLASS, Z_DIM, device=DEVICE)

    with torch.no_grad():
        fake_imgs = G(z, labels).cpu()

    # unnormalize [-1,1] -> [0,1]
    fake_imgs = (fake_imgs + 1) / 2

    # save images
    for i in range(GEN_PER_CLASS):
        save_path = os.path.join(out_dir, f"generated_{i+1}.jpg")
        save_image(fake_imgs[i], save_path)

    after = count_images(out_dir)
    print(f"Class {class_id}: Before = {before}, After = {after}")

print("\n🎉 Dataset successfully generated!")
print(f"Saved in: {OUTPUT_ROOT}")