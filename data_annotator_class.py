import os
import random

# Define data directory and target file paths
data_root = r'./data/PlantVillage'
train_file = 'train_class.txt'
val_file = 'val_class.txt'
test_file = 'test_class.txt'
label_file = 'labels_class.txt'

# Get subdirectories in the data root directory
subdirectories = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

# Create training, validation, and test files
with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f, open(test_file, 'w') as test_f:
    for class_index, subdirectory in enumerate(subdirectories):
        subdirectory_path = os.path.join(data_root, subdirectory)
        image_names = os.listdir(subdirectory_path)
        num_images = len(image_names)

        # Shuffle image names for random selection
        random.shuffle(image_names)

        # Determine the number of images for training, validation, and test
        num_train = int(0.8 * num_images)  # 80% for training
        num_val = int(0.1 * num_images)  # 10% for validation
        num_test = num_images - num_train - num_val  # Remaining 10% for test

        # Select images for training, validation, and test
        train_images = image_names[:num_train]
        val_images = image_names[num_train:num_train + num_val]
        test_images = image_names[num_train + num_val:]

        # Write image names and labels to the corresponding files
        for image_name in train_images:
            image_path = os.path.join(subdirectory_path, image_name)
            train_f.write(f"{image_path},{class_index}\n")

        for image_name in val_images:
            image_path = os.path.join(subdirectory_path, image_name)
            val_f.write(f"{image_path},{class_index}\n")

        for image_name in test_images:
            image_path = os.path.join(subdirectory_path, image_name)
            test_f.write(f"{image_path},{class_index}\n")

# Save labels to a file
with open(label_file, 'w') as label_f:
    label_f.write('\n'.join(subdirectories))
