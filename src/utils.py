import os
import shutil
import random

def prepare_classification_dataset(src_dir, dest_dir, split_ratio=0.8):
    """
    Prepares the dataset for YOLOv8-Classification.
    Structure:
    dest_dir/
        train/
            fire/
            non_fire/
        val/
            fire/
            non_fire/
    """
    classes = {
        'fire_images': 'fire',
        'non_fire_images': 'non_fire'
    }

    for split in ['train', 'val']:
        for cls_name in classes.values():
            os.makedirs(os.path.join(dest_dir, split, cls_name), exist_ok=True)

    for src_folder, cls_name in classes.items():
        src_path = os.path.join(src_dir, src_folder)
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Copy to train
        for img in train_images:
            shutil.copy(os.path.join(src_path, img), os.path.join(dest_dir, 'train', cls_name, img))
        
        # Copy to val
        for img in val_images:
            shutil.copy(os.path.join(src_path, img), os.path.join(dest_dir, 'val', cls_name, img))

    print(f"Dataset prepared at {dest_dir}")

if __name__ == "__main__":
    SRC = r"C:\Users\Vaibhav\Desktop\NP-3\fire_dataset"
    DEST = r"c:\Users\Vaibhav\Desktop\NP-3\data\processed"
    prepare_classification_dataset(SRC, DEST)