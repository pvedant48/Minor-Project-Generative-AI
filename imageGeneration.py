# get generated class-wise samples from latest checkpoint
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

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


# ===============================================
# CONFIG
# ===============================================
ROOT_DIR = "MOBIUS/Images"
SAVE_DIR = "Images/cDCGAN/results"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 256
NUM_CLASSES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================================
# TRANSFORM (same as training)
# ===============================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

inv_normalize = transforms.Normalize(
    mean=[-1, -1, -1],
    std=[2, 2, 2]
)

# ===============================================
# LOAD cDCGAN GENERATOR
# ===============================================
z_dim = 100
G = Generator(z_dim=z_dim, num_classes=NUM_CLASSES, img_channels=3).to(DEVICE)

ckpt = torch.load("savedModels/cDCGAN_MOBIUS/latest_checkpoint.pth", map_location=DEVICE)
G.load_state_dict(ckpt["generator_state_dict"])
G.eval()

print("✅ Loaded Generator Checkpoint")


# ===============================================
# HELPER — Load 8 REAL images for a class
# ===============================================
def load_real_images(class_id, n=8):
    class_path = os.path.join(ROOT_DIR, str(class_id))
    files = [os.path.join(class_path, f) for f in os.listdir(class_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_bad" not in f.lower()]

    files = sorted(files)[:n]  # take first 8 consistent images

    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img = transform(img)
        imgs.append(img)

    return torch.stack(imgs)  # shape: (8, 3, H, W)


# ===============================================
# HELPER — Generate 8 FAKE images for a class
# ===============================================
def generate_fake_images(class_id, n=8):
    z = torch.randn(n, z_dim, device=DEVICE)
    labels = torch.full((n,), class_id, dtype=torch.long, device=DEVICE)
    fake = G(z, labels).detach().cpu()

    return fake


# ===============================================
# HELPER — Plot real (left) & fake (right)
# ===============================================
def save_plot(real_imgs, fake_imgs, class_id):
    real_imgs = inv_normalize(real_imgs)
    fake_imgs = inv_normalize(fake_imgs)

    fig, axs = plt.subplots(2, 8, figsize=(20, 6))
    fig.suptitle(f"Class {class_id} — Real (Top) vs Fake (Bottom)", fontsize=18)

    # Real
    for i in range(8):
        axs[0, i].imshow(real_imgs[i].permute(1, 2, 0).clamp(0,1).numpy())
        axs[0, i].axis("off")

    # Fake
    for i in range(8):
        axs[1, i].imshow(fake_imgs[i].permute(1, 2, 0).clamp(0,1).numpy())
        axs[1, i].axis("off")

    save_path = os.path.join(SAVE_DIR, f"class_{class_id}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved → {save_path}")


# ===============================================
# MAIN LOOP — Generate 100 plots
# ===============================================
for class_id in tqdm(range(NUM_CLASSES), desc="Generating Plots"):
    real_imgs = load_real_images(class_id+1)
    fake_imgs = generate_fake_images(class_id+1)

    save_plot(real_imgs, fake_imgs, class_id+1)

print("\n🎉 All 100 plots generated successfully!")