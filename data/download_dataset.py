"""
Dataset acquisition, organization, and generation module for Plant Disease Detection.
Organizes data into train (70%), validation (15%), and test (15%) splits.
"""

import os
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# Default representative PlantVillage classes
DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
]


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split a directory of class folders into train, val, and test subdirectories.
    """
    random.seed(seed)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in source_dir.iterdir() if p.is_dir() and not p.name.startswith(('.', 'train', 'val', 'test'))]
    if not class_dirs:
        print(f"[!] No class subdirectories found in {source_dir}")
        return

    print(f"[*] Found {len(class_dirs)} classes in {source_dir}. Splitting...")

    for class_path in class_dirs:
        class_name = class_path.name
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
        images = [f for f in class_path.iterdir() if f.suffix in valid_exts]
        random.shuffle(images)

        n_total = len(images)
        if n_total == 0:
            continue

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :]

        for img in train_imgs:
            shutil.copy2(img, train_dir / class_name / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_dir / class_name / img.name)
        for img in test_imgs:
            shutil.copy2(img, test_dir / class_name / img.name)

        print(f"    Class '{class_name}': {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print("[+] Dataset split successfully completed!")


def generate_synthetic_leaf(class_name: str, width: int = 256, height: int = 256) -> Image.Image:
    """
    Synthesize realistic leaf texture and disease lesions.
    Healthy leaves feature chloroplast gradients, natural vein structures, and organic margins.
    Diseased leaves feature distinct spots, concentric rings (blights), rust pustules, and necrosis.
    """
    is_healthy = "healthy" in class_name.lower()
    
    # Background: natural neutral greenhouse/soil backdrop
    bg_color = (
        random.randint(30, 45),
        random.randint(30, 42),
        random.randint(28, 38)
    )
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Base leaf coloration
    if "corn" in class_name.lower():
        # Elongated monocot leaf
        leaf_color = (random.randint(60, 95), random.randint(140, 180), random.randint(40, 65))
        center_x, center_y = width // 2, height // 2
        leaf_points = [
            (center_x, 20),
            (center_x + random.randint(50, 75), center_y),
            (center_x + random.randint(20, 35), height - 20),
            (center_x - random.randint(20, 35), height - 20),
            (center_x - random.randint(50, 75), center_y),
        ]
    else:
        # Dicot broad leaf (Tomato, Apple, Potato)
        leaf_color = (random.randint(45, 80), random.randint(120, 175), random.randint(35, 60))
        center_x, center_y = width // 2, height // 2
        leaf_points = [
            (center_x, 25),
            (center_x + random.randint(65, 95), center_y - 20),
            (center_x + random.randint(50, 80), center_y + 60),
            (center_x, height - 30),
            (center_x - random.randint(50, 80), center_y + 60),
            (center_x - random.randint(65, 95), center_y - 20),
        ]

    # Draw leaf body
    draw.polygon(leaf_points, fill=leaf_color)

    # Leaf main vein & secondary veins
    vein_color = (min(255, leaf_color[0] + 25), min(255, leaf_color[1] + 30), min(255, leaf_color[2] + 20))
    draw.line([(center_x, 30), (center_x, height - 35)], fill=vein_color, width=3)
    for vy in range(60, height - 60, 25):
        draw.line([(center_x, vy), (center_x - random.randint(35, 65), vy - random.randint(10, 20))], fill=vein_color, width=2)
        draw.line([(center_x, vy), (center_x + random.randint(35, 65), vy - random.randint(10, 20))], fill=vein_color, width=2)

    # Add disease symptoms if not healthy
    if not is_healthy:
        c_lower = class_name.lower()
        if "scab" in c_lower or "black_rot" in c_lower:
            # Dark olivaceous/black circular lesions with chlorotic halos
            num_spots = random.randint(4, 9)
            for _ in range(num_spots):
                sx = center_x + random.randint(-55, 55)
                sy = center_y + random.randint(-65, 65)
                r = random.randint(10, 24)
                # Chlorotic halo (yellow rim)
                draw.ellipse([sx - r - 3, sy - r - 3, sx + r + 3, sy + r + 3], fill=(160, 150, 40))
                # Necrotic center (dark brown/black)
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(random.randint(25, 45), random.randint(20, 35), 15))
        elif "rust" in c_lower:
            # Orange-brown powdery pustules
            num_pustules = random.randint(15, 30)
            for _ in range(num_pustules):
                sx = center_x + random.randint(-50, 50)
                sy = center_y + random.randint(-70, 70)
                r = random.randint(4, 9)
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(random.randint(190, 230), random.randint(85, 120), 20))
        elif "blight" in c_lower:
            # Concentric rings / target-like brown lesions or water-soaked patches
            num_patches = random.randint(3, 6)
            for _ in range(num_patches):
                sx = center_x + random.randint(-50, 50)
                sy = center_y + random.randint(-60, 60)
                r = random.randint(16, 32)
                # Outer ring
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(110, 75, 35))
                # Inner ring
                draw.ellipse([sx - r + 5, sy - r + 5, sx + r - 5, sy + r - 5], fill=(70, 45, 20))
                # Center
                draw.ellipse([sx - r + 10, sy - r + 10, sx + r - 10, sy + r - 10], fill=(40, 25, 10))
        elif "bacterial_spot" in c_lower or "septoria" in c_lower:
            # Small angular brown spots with yellow halos
            num_spots = random.randint(18, 35)
            for _ in range(num_spots):
                sx = center_x + random.randint(-55, 55)
                sy = center_y + random.randint(-65, 65)
                r = random.randint(3, 7)
                draw.ellipse([sx - r - 2, sy - r - 2, sx + r + 2, sy + r + 2], fill=(180, 170, 50))
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(50, 30, 20))
        elif "mold" in c_lower:
            # Pale greenish-yellow spots transitioning to velvety olive-brown patches
            num_spots = random.randint(5, 10)
            for _ in range(num_spots):
                sx = center_x + random.randint(-50, 50)
                sy = center_y + random.randint(-60, 60)
                r = random.randint(12, 22)
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(140, 130, 40))

    # Apply subtle realistic smoothing
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def create_sample_dataset(
    output_dir: Path,
    samples_per_class: int = 40,
    classes=None,
    seed: int = 42,
):
    """
    Generate a balanced sample dataset structured into train, val, and test splits.
    Ensures complete immediate reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    if classes is None:
        classes = DEFAULT_CLASSES

    raw_dir = output_dir / "_raw_temp"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating {samples_per_class} realistic sample leaves for {len(classes)} classes...")
    for idx, cname in enumerate(classes):
        c_folder = raw_dir / cname
        c_folder.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = generate_synthetic_leaf(cname, width=256, height=256)
            img.save(c_folder / f"{cname}_{i:03d}.jpg", quality=92)
        print(f"    [{idx+1}/{len(classes)}] Created {cname}")

    # Now split into train, val, test
    split_dataset(raw_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)

    # Clean up raw temp
    shutil.rmtree(raw_dir, ignore_errors=True)
    print(f"[+] Sample dataset successfully created at {output_dir}")


def export_sample_images_for_demo(output_dir: Path, demo_dir: Path, n_per_class: int = 1):
    """
    Export representative test leaves to sample_images/ for instant demo in Streamlit app.
    """
    demo_dir.mkdir(parents=True, exist_ok=True)
    test_dir = output_dir / "test"
    if not test_dir.exists():
        return

    for c_dir in test_dir.iterdir():
        if c_dir.is_dir():
            imgs = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png"))
            if imgs:
                for idx, sample in enumerate(imgs[:n_per_class]):
                    dest = demo_dir / f"{c_dir.name}_sample{idx+1}.jpg"
                    shutil.copy2(sample, dest)
    print(f"[+] Exported demo sample images to {demo_dir}")


def main():
    parser = argparse.ArgumentParser(description="PlantVillage Dataset Setup & Organizer")
    parser.add_argument("--data_dir", type=str, default="./data", help="Target dataset root directory")
    parser.add_argument("--source_dir", type=str, default=None, help="Path to raw PlantVillage folder if downloaded manually")
    parser.add_argument("--samples_per_class", type=int, default=45, help="Number of samples per class for built-in generation")
    parser.add_argument("--export_demo", action="store_true", default=True, help="Export demo sample images for Streamlit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.source_dir and Path(args.source_dir).exists():
        print(f"[*] Splitting user-provided dataset from: {args.source_dir}")
        split_dataset(Path(args.source_dir), data_dir)
    else:
        # Check if train/val/test already populated
        train_path = data_dir / "train"
        if train_path.exists() and any(train_path.iterdir()):
            print(f"[+] Existing dataset found in {data_dir}. Skipping generation.")
        else:
            print("[*] Creating high-fidelity representative PlantVillage dataset...")
            create_sample_dataset(data_dir, samples_per_class=args.samples_per_class)

    if args.export_demo:
        demo_dir = data_dir.parent / "sample_images"
        export_sample_images_for_demo(data_dir, demo_dir)


if __name__ == "__main__":
    main()
