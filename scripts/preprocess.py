import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import transforms

def split_data(data_dir, output_dir, test_size=0.2):
    """
    Splits images into train and validation folders.
    """
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

    all_images = [f for f in os.listdir(data_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    train_images, val_images = train_test_split(all_images, test_size=test_size, random_state=42)

    for img in train_images:
        shutil.copy(os.path.join(data_dir, img), os.path.join(output_dir, "train", img))

    for img in val_images:
        shutil.copy(os.path.join(data_dir, img), os.path.join(output_dir, "val", img))

    print(f"Train images: {len(train_images)}, Validation images: {len(val_images)}")

if __name__ == "__main__":
    split_data("./data/images", "./data/processed")

