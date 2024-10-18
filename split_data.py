# Used to split the miniImagenet into Val and Train

import os
import shutil
import random



dataset_path = './filelists/miniImagenet/'
train_path = './filelists/miniImagenet/train'
val_path = './filelists/miniImagenet/val'



# Create directories for train and validation sets
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Iterate through each class directory
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    
    if os.path.isdir(class_dir):  # Ensure it's a directory
        images = [img for img in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, img))]
        
        if len(images) == 0:
            print(f"No images found in {class_dir}. Skipping.")
            continue
        
        random.shuffle(images)  # Shuffle images for randomness
        
        # Calculate split index
        split_index = int(0.8 * len(images))
        
        # Split images into train and val sets
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        # Create class directories in train and val paths
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
        
        # Move images to respective directories
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_path, class_name, img)
            shutil.copy(src, dst)
        
        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_path, class_name, img)
            shutil.copy(src, dst)

print("Dataset split completed.")